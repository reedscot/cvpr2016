-- Necessary functionalities
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'

local model_utils = require('util.model_utils')

cutorch.setDevice(1)

-- Encode query document using alphabet.
local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end

-------------------------------------------------
cmd = torch.CmdLine()
cmd:option('-data_dir','data','data directory.')
cmd:option('-image_dir','images_th3','image subdirectory.')
cmd:option('-txt_dir','','text subdirectory.')
cmd:option('-savefile','sje_tcnn','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-symmetric',1,'symmetric sje')
cmd:option('-learning_rate',0.0001,'learning rate')
cmd:option('-testclasses', 'testclasses.txt', 'validation or test classes to be used in evaluation')
cmd:option('-ids_file', 'trainvalids.txt', 'file specifying which class labels were used for training.')
cmd:option('-model','','model to load. If blank then above options will be used.')
cmd:option('-txt_limit',0,'if 0 then use all available text. Otherwise limit the number of documents per class')
cmd:option('-num_caption',10,'number of captions per image to be used for training')
cmd:option('-ttype','char','word|char')
cmd:option('-gpuid',0,'gpu to use')

opt = cmd:parse(arg)
local model
if opt.model ~= '' then
	model = torch.load(opt.model)
else
	model = torch.load(string.format('%s/lm_%s_%.5f_%.0f_%.0f_%s.t7', opt.checkpoint_dir, opt.savefile, opt.learning_rate, opt.symmetric, opt.num_caption, opt.ids_file))
end
-----------------------------------------------------------

if opt.gpuid >= 0 then
  cutorch.setDevice(opt.gpuid+1)
end

local doc_length = model.opt.doc_length
local protos = model.protos
protos.enc_doc:evaluate()
protos.enc_image:evaluate()

function extract_img(filename)
    local data = torch.load(filename)
    if data:size():size() == 3 then
        local fea = data[{{},{},1}]
        fea = fea:float():cuda()
        local out = protos.enc_image:forward(fea)
        return out:float()
    else
        local fea = data[{{},{},{},{},1}]
        fea = fea:float():cuda()
        local out = protos.enc_image:forward(fea)
        return out:float()
    end
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
    --local txt = torch.load(filename):permute(1,3,2):add(1)
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
    local out = protos.enc_doc:forward(txt_mat)
    out = torch.mean(out,1):float()
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
    local out = protos.enc_doc:forward(txt_mat)
    return torch.mean(out,1):float()
end

function classify(txt_dir, img_dir, cls_list)
    local acc = 0.0
    local total = 0.0
    local fea_img = {}
    local fea_txt = {}
    for fname in io.lines(cls_list) do
        local imgpath = img_dir .. '/' .. fname .. '.t7'
        local txtpath = txt_dir .. '/' .. fname .. '.t7'
        fea_img[#fea_img + 1] = extract_img(imgpath)
        fea_txt[#fea_txt + 1] = extract_txt(txtpath)
    end
    for i = 1,#fea_img do
        -- loop over individual images.
        for k = 1,fea_img[i]:size(1) do
            local best_match = 1
            local best_score = -math.huge
            for j = 1,#fea_txt do
                local cur_score = torch.dot(fea_img[i][{k,{}}], fea_txt[j])
                if cur_score > best_score then
                    best_match = j
                    best_score = cur_score
                end
            end
            if best_match == i then
                acc = acc + 1
            end
            total = total + 1
        end
    end
    return acc / total
end

local txt_dir
if opt.txt_dir == '' then
    if opt.ttype == 'char' then
        txt_dir = string.format('%s/text_c%d', opt.data_dir, opt.num_caption)
    else -- word
        txt_dir = string.format('%s/word_c%d', opt.data_dir, opt.num_caption)
    end
else
    txt_dir = string.format('%s/%s', opt.data_dir, opt.txt_dir)
end
if opt.ttype == 'word' then
    vocab_size = 0
    for k,v in pairs(model.vocab) do
        vocab_size = vocab_size + 1
    end
end
local img_dir = string.format('%s/%s', opt.data_dir, opt.image_dir)
local testcls = string.format('%s/%s', opt.data_dir, opt.testclasses)
local test_acc     = classify(txt_dir, img_dir, testcls)
print(string.format('Average top-1 val/test accuracy: %6.4f\n', test_acc))

