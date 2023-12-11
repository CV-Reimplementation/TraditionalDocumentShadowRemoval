function [out] = matching(tgt,src)
% Matching average color of src and target
temp = src;
m_src_bg = reshape(mean(mean(src)),1,3);
m_tgt_bg = reshape(mean(mean(tgt)),1,3);
temp(:,:,1) = temp(:,:,1)*(m_tgt_bg(1)/m_src_bg(1));
temp(:,:,2) = temp(:,:,2)*(m_tgt_bg(2)/m_src_bg(2));
temp(:,:,3) = temp(:,:,3)*(m_tgt_bg(3)/m_src_bg(3));
out = temp;
end