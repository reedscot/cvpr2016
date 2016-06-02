
function rank_images(fea_txt, fea_id, fea_imgs, scorefunc, attfunc)
    attfunc = attfunc or (function(a,b) return 0 end)
    -- find total number of images.
    local total_imgs = 0
    for i = 1,#fea_imgs do
        total_imgs = total_imgs + fea_imgs[i]:size(1)
    end

    -- compute scores and matches.
    local scores = torch.zeros(total_imgs)
    local matches = torch.zeros(total_imgs)
    local cosine = torch.zeros(total_imgs)
    local img_ix = 1
    for i = 1,#fea_imgs do
        for j = 1,fea_imgs[i]:size(1) do
            local score = scorefunc(fea_imgs[i][{j,{}}], fea_txt)
            scores[img_ix] = score
            if i == fea_id then
                matches[img_ix] = 1
            else
                matches[img_ix] = 0
            end
            cosine[img_ix] = attfunc(i, fea_id)
            img_ix = img_ix + 1
        end
    end

    scores, ix = torch.sort(scores,1,true)
    local tmp = matches:clone()
    local tmp2 = cosine:clone()
    for i = 1,matches:size(1) do
        matches[i] = tmp[ix[i] ]
        cosine[i] = tmp2[ix[i] ]
    end

    return { scores, matches, cosine }
end

function retrieval_at_k(txt_dir, img_dir, cls_list, kvals, scorefunc, limit)
    local fea_img = {}
    local fea_txt = {}

    --local total_txt = 0
    local total_img = 0
    for fname in io.lines(cls_list) do
        local imgpath = img_dir .. '/' .. fname .. '.t7'
        local txtpath = txt_dir .. '/' .. fname .. '.t7'

        local tmp_img = extract_img(imgpath)
        fea_img[#fea_img + 1] = tmp_img

        local tmp_txt
        if limit ~= nil then
            tmp_txt = extract_txt(txtpath, limit)
        else
            tmp_txt = extract_txt(txtpath)
        end
        fea_txt[#fea_txt + 1] = tmp_txt

        total_img = total_img + tmp_img:size(1)
    end
    local total_txt = #fea_txt

    -- rank images for each description.
    scores = torch.zeros(#fea_txt, total_img)
    matches = torch.zeros(#fea_txt, total_img)
    local min_score = math.huge
    local max_score = -math.huge
    for i = 1,#fea_txt do
        local res = rank_images(fea_txt[i], i, fea_img, scorefunc)
        scores[{i,{}}] = res[1]
        matches[{i,{}}] = res[2]
    end
    local results = {}
    for _,k in ipairs(kvals) do
        results[k] = torch.mean(matches:narrow(2,1,k))
    end
    return results
end
