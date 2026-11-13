-- PDF 内的本地文档链接指向公开仓库，避免链接依赖作者电脑。
function Link(el)
  if not el.target:match("^%a[%w+.-]*:") and not el.target:match("^#") then
    local target = el.target
    if target:sub(1, 3) == "../" then
      target = target:sub(4)
    else
      target = "reports/" .. target
    end
    el.target = "https://github.com/ZureTz/ml-network-intrusion-detection/blob/master/" .. target
  end
  return el
end

function Image(el)
  local pdf = el.src:gsub("%.png$", ".pdf")
  local f = io.open(pdf, "rb")
  if f then f:close(); el.src = pdf end
  el.attributes.width = "95%"
  return el
end

-- 所有层级保持常规字重。
function Strong(el) return el.content end

function Code(el)
  if #el.text > 30 and not el.text:match("[{}%s]") then
    return pandoc.RawInline("latex", "\\nolinkurl{" .. el.text .. "}")
  end
end

function Header(el)
  local title = pandoc.utils.stringify(el.content)
  if title == "参考文献" then
    return {pandoc.RawBlock("latex", "\\clearpage"), el}
  elseif title == "混淆矩阵与攻击类别错误" then
    return {pandoc.RawBlock("latex", "\\Needspace{12\\baselineskip}"), el}
  end
end

function Table(el)
  local rows = #el.head.rows + #el.foot.rows
  for _, body in ipairs(el.bodies) do rows = rows + #body.body + #body.head end
  return {pandoc.RawBlock("latex", "\\Needspace{" .. math.min(rows + 2, 17) .. "\\baselineskip}"), el}
end
