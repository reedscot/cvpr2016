require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a multi-modal embedding model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/cub_c10','data directory.')
cmd:option('-ids_file', 'trainids.txt', 'file specifying which class labels are used for training. Can also be trainvalids.txt')
cmd:option('-batch_size',40,'number of sequences to train on in parallel')
cmd:option('-image_dim',1024,'image feature dimension')
cmd:option('-emb_dim',1536,'embedding dimension')
cmd:option('-image_noop',1,'if 1, the image encoder is a no-op. In this case emb_dim and image_dim must match.')
cmd:option('-randomize_pair',0,'if 1, images and captions of the same class are randomly paired.')
cmd:option('-doc_length',201,'document length')
cmd:option('-nclass',200,'number of classes')
cmd:option('-dropout',0.0,'dropout rate')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-savefile','sje_hybrid','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-max_epochs',300,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-learning_rate',0.0004,'learning rate')
cmd:option('-learning_rate_decay',0.98,'learning rate decay')
cmd:option('-learning_rate_decay_after',1,'in number of epochs, when to start decaying the learning rate')
cmd:option('-print_every',100,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-symmetric',1,'whether to use symmetric form of SJE')
cmd:option('-num_caption',5,'number of captions per image to be used for training')
cmd:option('-image_dir','images_th3','image directory in data')
cmd:option('-flip',0,'flip sentence')
cmd:option('-bidirectional',0,'use bidirectional version')
cmd:option('-avg', 0, 'whether to time-average hidden units')
cmd:option('-cnn_dim', 256, 'char-cnn embedding dimension')

opt = cmd:parse(arg)
if opt.image_noop then
  opt.emb_dim = opt.image_dim
end
torch.manualSeed(opt.seed)
print(opt)

if opt.bidirectional == 1 then
  FixedRNN = require('modules.BidirectionalRNN')
else
  FixedRNN = require('modules.FixedRNN')
end
DocumentCNN = require('modules.HybridCNN')
local ImageEncoder = require('modules.ImageEncoder')
local MultimodalMinibatchLoader = require('util.MultimodalMinibatchLoaderCaption')
local model_utils = require('util.model_utils')


-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

local loader = MultimodalMinibatchLoader.create(
    opt.data_dir, opt.nclass, opt.image_dim, opt.doc_length,
    opt.batch_size, opt.randomize_pair, opt.ids_file, opt.num_caption,
    opt.image_dir, opt.flip)

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

local do_random_init = false
if string.len(opt.init_from) > 0 then
    print('loading from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
else
    protos = {}
    protos.enc_image = ImageEncoder.enc(opt.image_dim, opt.emb_dim, opt.image_noop)
    protos.enc_doc = DocumentCNN.cnn(loader.alphabet_size, opt.emb_dim, opt.dropout, opt.avg, opt.cnn_dim)
    protos.enc_image:training()
    protos.enc_doc:training()
    do_random_init = true
end

if opt.gpuid >= 0 then
    for k,v in pairs(protos) do
        if v.weights ~= nil then
            v.weights = v.weights:float():cuda()
            v.grads = v.grads:float():cuda()
        else
            v:cuda()
        end
    end
end
params, grad_params = model_utils.combine_all_parameters(protos.enc_image, protos.enc_doc)
if do_random_init then
    params:uniform(-0.08, 0.08) -- small numbers uniform
end

acc_batch = 0.0
acc_smooth = 0.0
function JointEmbeddingLoss(fea_txt, fea_img, labels)
    local batch_size = fea_img:size(1)
    local num_class = fea_txt:size(1)
    local score = torch.zeros(batch_size, num_class)
    local txt_grads = fea_txt:clone():fill(0)
    local img_grads = fea_img:clone():fill(0)

    local loss = 0
    acc_batch = 0.0
    for i = 1,batch_size do
        for j = 1,num_class do
            score[{i,j}] = torch.dot(fea_img:narrow(1,i,1), fea_txt:narrow(1,j,1))
        end
        local label_score = score[{i,labels[i]}]
        for j = 1,num_class do
            if j ~= labels[i] then
                local cur_score = score[{i,j}]
                local thresh = cur_score - label_score + 1
                if (thresh > 0) then
                    loss = loss + thresh
                    local txt_diff = fea_txt:narrow(1,j,1) - fea_txt:narrow(1,labels[i],1)
                    img_grads:narrow(1, i, 1):add(txt_diff)
                    txt_grads:narrow(1, j, 1):add(fea_img:narrow(1,i,1))
                    txt_grads:narrow(1, labels[i], 1):add(-fea_img:narrow(1,i,1))
                end
            end 
        end
        local max_score, max_ix = score:narrow(1,i,1):max(2)
        if (max_ix[{1,1}] == labels[i]) then
            acc_batch = acc_batch + 1
        end
    end
    acc_batch = 100 * (acc_batch / batch_size)
    local denom = batch_size * num_class
    local res = { [1] = txt_grads:div(denom),
                  [2] = img_grads:div(denom) }
    acc_smooth = 0.99 * acc_smooth + 0.01 * acc_batch
    return loss / denom, res
end


-- check embedding gradient.
function wrap_emb(inp, nh, nx, ny, labs)
    local x = inp:narrow(1,1,nh*nx):clone():reshape(nx,nh)
    local y = inp:narrow(1,nh*nx + 1,nh*ny):clone():reshape(ny,nh)
    local loss, grads = JointEmbeddingLoss(x, y, labs)
    local dx = grads[1]
    local dy = grads[2]
    local grad = torch.cat(dx:reshape(nh*nx), dy:reshape(nh*ny))
    return loss, grad
end
if opt.checkgrad == 1 then
    print('\nChecking embedding gradient\n')
    local nh = 3
    local nx = 4
    local ny = 2
    local txt = torch.randn(nx, nh)
    local img = torch.randn(ny, nh)
    local labs = torch.randperm(nx)
    local initpars = torch.cat(txt:clone():reshape(nh*nx), img:clone():reshape(nh*ny))
    local opfunc = function(curpars) return wrap_emb(curpars, nh, nx, ny, labs) end
    diff, dC, dC_est = checkgrad(opfunc, initpars, 1e-3)
    print(dC)
    print(dC_est)
    print(diff)
    debug.debug()
end


function feval_wrap(pars)
    ------------------ get minibatch -------------------
    local txt, img, labels = loader:next_batch()
    return feval(pars, txt, img, labels)
end

function feval(newpars, txt, img, labels)
    if newpars ~= params then
        params:copy(newpars)
    end
    grad_params:zero()

    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        txt = txt:float():cuda()
        img = img:float():cuda()
        labels = labels:float():cuda()
    end
    ------------------- forward pass -------------------
    local fea_txt = protos.enc_doc:forward(txt)
    local fea_img = protos.enc_image:forward(img)

    -- Criterion --
    local loss, grads = JointEmbeddingLoss(fea_txt, fea_img, labels)
    local dtxt = grads[1]       -- backprop through document CNN.
    local dimg = grads[2]       -- backprop through image encoder.

    if opt.symmetric == 1 then
        local loss2, grads2 = JointEmbeddingLoss(fea_img, fea_txt, labels)
        dtxt:add(grads2[2])       -- backprop through document CNN.
        dimg:add(grads2[1])       -- backprop through image encoder.
        loss = loss + loss2
    end

    protos.enc_doc:backward(txt, dtxt)
    protos.enc_image:backward(img, dimg)

    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval_wrap, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = 0
        val_losses[i] = val_loss

      	local savefile  = string.format('%s/lm_%s_%.5f_%.0f_%.0f_%s.t7', opt.checkpoint_dir, opt.savefile, opt.learning_rate, opt.symmetric, opt.num_caption, opt.ids_file)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (ep %.3f), loss=%4.2f, acc1=%4.2f, acc2=%.4f, g/p=%6.4e, t/b=%.2fs",
              i, iterations, epoch, train_loss, acc_batch, acc_smooth, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss0 == nil then loss0 = loss[1] end
    if ((loss[1] > loss0 * 3) and (i > 10)) then
        print('loss is exploding, aborting.')
        break -- halt
    end
end

