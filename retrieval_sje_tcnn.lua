-- Retrieve images based on captions.

require('nn')
require('nngraph')
require('cutorch')
require('cunn')
require('cudnn')
require('util.util_retrieval')

local model_utils = require('util.model_utils')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end

-------------------------------------------------
cmd = torch.CmdLine()
cmd:option('-data_dir','data','data directory.')
cmd:option('-savefile','sje_tcnn','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-symmetric',1,'symmetric sje')
cmd:option('-learning_rate',0.0001,'learning rate')
cmd:option('-testclasses', 'testclasses.txt', 'validation or test classes to be used in evaluation')
cmd:option('-ids_file', 'trainvalids.txt', 'file specifying which class labels were used for training.')
cmd:option('-model','','model to load. If blank then above options will be used.')
cmd:option('-txt_limit',0,'if 0 then use all available text. Otherwise limit the number of documents per class')
cmd:option('-num_caption',10,'numner of captions per image to be used for training')
cmd:option('-outfile', 'results/roc.csv', 'output csv file with ROC curves.')
cmd:option('-ttype','char','word|char')

opt = cmd:parse(arg)
local model
if opt.model ~= '' then
	model = torch.load(opt.model)
else
	model = torch.load(string.format('%s/lm_%s_%.5f_%.0f_%.0f_%s.t7', opt.checkpoint_dir, opt.savefile, opt.learning_rate, opt.symmetric, opt.num_caption, opt.ids_file))
end
-----------------------------------------------------------

local doc_length = model.opt.doc_length
local protos = model.protos
protos.enc_doc:evaluate()
protos.enc_image:evaluate()
--print(model.opt)

function extract_img(filename)
    local fea = torch.load(filename)[{{},{},1}]
    fea = fea:float():cuda()
    local out = protos.enc_image:forward(fea):clone()
    return out:cuda()
end

function extract_txt(filename)
    if opt.ttype == 'word' then
        return extract_txt_word(filename)
    else -- char
        return extract_txt_char(filename)
    end
end

function extract_txt_word(filename)
    -- average all text features together.
    local txt = torch.load(filename):permute(1,3,2)
    txt = txt:reshape(txt:size(1)*txt:size(2),txt:size(3)):float():cuda()
    if opt.txt_limit > 0 then
        local actual_limit = math.min(txt:size(1), opt.txt_limit)
        txt_order = torch.randperm(txt:size(1)):sub(1,actual_limit)
        local tmp = txt:clone()
        for i = 1,actual_limit do
            txt[{i,{}}]:copy(tmp[{txt_order[i],{}}])
        end
        txt = txt:narrow(1,1,actual_limit)
    end

    if (model.opt.num_repl ~= nil) then
        tmp = txt:clone()
        txt = torch.ones(txt:size(1),model.opt.num_repl*txt:size(2))
        for i = 1,txt:size(1) do
            local cur_sen = torch.squeeze(tmp[{i,{}}]):clone()
            local cur_len = cur_sen:size(1) - cur_sen:eq(1):sum()
            local txt_ix = 1
            for j = 1,cur_len do
                for k = 1,model.opt.num_repl do
                    txt[{i,txt_ix}] = cur_sen[j]
                    txt_ix = txt_ix + 1
                end
            end
        end
    end

    local txt_mat = torch.zeros(txt:size(1), txt:size(2), vocab_size+1)
    for i = 1,txt:size(1) do
        for j = 1,txt:size(2) do
            local on_ix = txt[{i, j}]
            if on_ix == 0 then
                break
            end
            txt_mat[{i, j, on_ix}] = 1
        end
    end
    txt_mat = txt_mat:float():cuda()
    local out = protos.enc_doc:forward(txt_mat):clone()
    out = torch.mean(out,1)
    return out
end

function extract_txt_char(filename)
    -- average all text features together.
    local txt = torch.load(filename):permute(1,3,2)
    txt = txt:reshape(txt:size(1)*txt:size(2),txt:size(3)):float():cuda()
    if opt.txt_limit > 0 then
        local actual_limit = math.min(txt:size(1), opt.txt_limit)
        txt_order = torch.randperm(txt:size(1)):sub(1,actual_limit)
        local tmp = txt:clone()
        for i = 1,actual_limit do
            txt[{i,{}}]:copy(tmp[{txt_order[i],{}}])
        end
        txt = txt:narrow(1,1,actual_limit)
    end
    local txt_mat = torch.zeros(txt:size(1), txt:size(2), #alphabet)
    for i = 1,txt:size(1) do
        for j = 1,txt:size(2) do
            local on_ix = txt[{i, j}]
            if on_ix == 0 then
                break
            end
            txt_mat[{i, j, on_ix}] = 1
        end
    end
    txt_mat = txt_mat:float():cuda()
    local out = protos.enc_doc:forward(txt_mat):clone()
    return torch.mean(out,1)
end

local txt_dir
if opt.ttype == 'char' then
    txt_dir = string.format('%s/text_c%d', opt.data_dir, opt.num_caption)
else -- word
    txt_dir = string.format('%s/word_c%d', opt.data_dir, opt.num_caption)
    vocab_size = 0
    for k,v in pairs(model.vocab) do
        vocab_size = vocab_size + 1
    end
end
local img_dir = string.format('%s/images', opt.data_dir)
local testcls = string.format('%s/%s', opt.data_dir, opt.testclasses)

local kvals = { 1, 5, 10, 50 }
local res = retrieval_at_k(txt_dir, img_dir, testcls, kvals, torch.dot, 1)
--print(string.format('mAP@1: %6.4f\n', res[1]))
--print(string.format('mAP@5: %6.4f\n', res[5]))
--print(string.format('mAP@10: %6.4f\n', res[10]))
print(string.format('mAP@50: %6.4f\n', res[50]))

--file = io.open(opt.outfile, "w")
--io.output(file)
io.write(string.format('%.4f,%.4f,%.4f,%.4f\n',
                       res[1],
                       res[5],
                       res[10],
                       res[50]))
io.close(file)

