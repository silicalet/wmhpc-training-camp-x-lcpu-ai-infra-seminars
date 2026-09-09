-- pandoc Lua filter:md 源里的结构标记 → 版式宏
--
-- 约定(见 handout/README.md):
--   ### 4.1 {.prob type=EXPERIMENT file=cuda/m4_gemm/05_thin_gemm.cu}
--     → \prob{4.1}{EXPERIMENT}[cuda/m4\_gemm/05\_thin\_gemm.cu]
--     选做加 opt=Optional;file 可省略
--   ::: reading / ::: lookback / ::: answer  → 对应 tcolorbox
--   ::: {.capstone title="prob 3.5(FROM-SCRATCH):block 内归约"}
--     → capstone 框

-- 先处理外层，避免可分页作答框嵌在另一个 tcolorbox 内而被截断。
traverse = 'topdown'

local function tex_escape(s)
  return (s:gsub("[%%#$&_{}]", "\\%0"))
end

local function latex_code(s)
  local tex = pandoc.write(pandoc.Pandoc({pandoc.Plain({pandoc.Code(s)})}), "latex")
  tex = tex:gsub("\\_", "\\_\\allowbreak{}")
  tex = tex:gsub("([/.,:=%-])", "%1\\allowbreak{}")
  return pandoc.RawInline("latex", tex)
end

function Code(c)
  return latex_code(c.text)
end

function Str(s)
  if s.text:find("_", 1, true) then
    return latex_code(s.text)
  end
end

function Header(h)
  if h.classes:includes("prob") then
    local num = pandoc.utils.stringify(h.content)
    local typ = h.attributes["type"] or "?"
    local opt = h.attributes["opt"]
    local file = h.attributes["file"]
    local cmd = "\\prob"
    if opt then cmd = cmd .. "[" .. opt .. "]" end
    cmd = cmd .. "{" .. num .. "}{" .. typ .. "}"
    if file then cmd = cmd .. "[" .. tex_escape(file) .. "]" end
    return pandoc.RawBlock("latex", cmd)
  end
end

local box_envs = { reading = true, lookback = true, answer = true }

function Div(d)
  for name in pairs(box_envs) do
    if d.classes:includes(name) then
      local blocks = pandoc.List({pandoc.RawBlock("latex", "\\begin{" .. name .. "}")})
      for _, block in ipairs(d.content) do
        local rows = 0
        if name == "answer" and block.t == "Table" then
          for _, body in ipairs(block.bodies) do rows = rows + #body.body end
        end
        -- 长表交给 longtable 分页，避免 tcolorbox 切页吞掉重复表头。
        if rows > 20 then
          blocks:insert(pandoc.RawBlock("latex", "\\end{" .. name .. "}"))
          blocks:insert(block)
          blocks:insert(pandoc.RawBlock("latex", "\\begin{" .. name .. "}"))
        else
          blocks:insert(block)
        end
      end
      blocks:insert(pandoc.RawBlock("latex", "\\end{" .. name .. "}"))
      return blocks
    end
  end
  if d.classes:includes("capstone") then
    local title = tex_escape(d.attributes["title"] or "")
    local file = d.attributes["file"]
    if file then
      title = title .. " \\hfill {\\footnotesize\\ttfamily " ..
        tex_escape(file) .. "}"
    end
    local blocks = pandoc.List()
    local opened = false
    for _, block in ipairs(d.content) do
      if block.t == "Div" and block.classes:includes("answer") then
        if opened then
          blocks:insert(pandoc.RawBlock("latex", "\\end{capstone}"))
          opened = false
        end
        blocks:insert(pandoc.Div({block}))
      else
        if not opened then
          blocks:insert(pandoc.RawBlock("latex", "\\begin{capstone}{" .. title .. "}"))
          opened = true
        end
        blocks:insert(block)
      end
    end
    if opened then
      blocks:insert(pandoc.RawBlock("latex", "\\end{capstone}"))
    end
    return blocks
  end
end
