# LaTeX 论文模板项目分析

这份文档详细分析了 `LaTeX-in-ICM-MCM/Template` 项目的组成结构和内容。该项目是一个用于数学建模竞赛（如 MCM/ICM）的 LaTeX 论文模板，包含完整的章节结构、代码排版、图表示例等。

## 1. 核心文件结构

项目主要由一个主文件 `main.tex` 和多个分章节的 `.tex` 文件组成，通过 `\input` 命令组合在一起。

| 文件名 | 对应章节/功能 | 关键内容 |
| :--- | :--- | :--- |
| **`main.tex`** | **主控文件** | 宏包加载、文章元数据（标题、队伍号）、摘要、目录、各章节整合 |
| **`part_1_pre.tex`** | **前言与准备** | 问题背景、重述、假设、符号说明、数据概览与预处理 |
| **`part_2_model.tex`** | **模型构建(示例)** | 公式排版、定理定义、伪代码算法、特殊图表布局示例 |
| **`part_3_conclusion.tex`** | **结果与结论** | 结论总结、灵敏度分析（TikZ绘图）、优缺点评价、模型扩展 |
| **`part_4_Appendix.tex`** | **附录** | 参考文献、代码清单（MATLAB/Python） |
| `easymcm.sty` | 样式文件 | 定义了模板的核心样式、页眉页脚、标题格式等（包含在Template目录下） |

---

## 2. 详细内容解读

### 2.1 主文件 (`main.tex`)
这是编译的入口文件。
- **排版设置**：加载了大量宏包（如图表 `graphicx`、代码高亮 `listings`、数学公式 `amsmath` 等）。
- **基本信息**：定义了标题 `\title`、摘要 `abstract` 和关键词。
- **结构控制**：按顺序引入了 Pre, Model, Conclusion, Appendix 四个部分。

### 2.2 第一部分：前言 (`part_1_pre.tex`)
这部分展示了论文开头的标准写法：
- **Introduction (引言)**：
  - `Problem Background`：展示了如何并排插入子图 (`subfigure`)。
  - `Restatement of the Problem`：使用了无序列表 (`itemize`) 列出问题重述。
  - 流程图：展示了如何插入跨栏大图。
- **Assumptions (假设)**：使用了带有自定义标签的枚举列表 (`enumerate`)。
- **Notations (符号说明)**：展示了一个标准的三线表 (`tabular` + `booktabs`)，包含符号、描述和单位。
- **Data Overview (数据准备)**：
  - 展示了缺失数据表。
  - 数据来源表 (`Data and Database Websites`)。
  - **长表格示例**：`Detailed Parameter Analysis` 展示了如何使用 `longtable` 进行跨页表格排版。

### 2.3 第二部分：模型排版示例 (`part_2_model.tex`)
这部分主要作为**“排版手册”**，教用户如何撰写数学内容：
- **公式 (Formulas)**：
  - 定义 (`Definition`)、定理 (`Theorem`)、例子 (`Example`) 的环境使用。
  - 复杂矩阵和多行公式 (`align`) 的排版。
- **图表 (Figures)**：
  - 展示了三张图并排 (`subfigure`)。
  - 使用 `minipage` 实现图片的并排布局。
- **算法 (Algorithms)**：
  - 包含了 `CAT Fusion Pseudo-code`，展示了如何使用 `algorithm2e` 宏包编写伪代码。

### 2.4 第三部分：结论与分析 (`part_3_conclusion.tex`)
- **Conclusions (结论)**：结论列表。
- **Sensitivity Analysis (灵敏度分析)**：
  - **TikZ 绘图**：代码中直接画了一个三维坐标系 (`tikzpicture`)，展示了 LaTeX 强大的绘图能力。
  - 可视化分析：展示了四张子图 (`2x2` grid) 的排版方式。
- **Model Evaluation (评价)**：模型优缺点分析。
- **图文混排**：展示了 `wrapfigure` 用法，让文字环绕图片。
- **特色排版**：
  - 首字下沉 (`lettrine`)。
  - 花体字引用 (`Author's Words`)。

### 2.5 第四部分：附录 (`part_4_Appendix.tex`)
- **参考文献 (`thebibliography`)**：列出了文献引用的标准格式。
- **代码展示**：
  - **MATLAB 代码**：使用了自定义的 `matlab` 环境（基于 `tcolorbox`），带有行号和背景色。
  - **Python 代码**：使用了自定义的 `python` 环境，展示了不同的代码高亮风格。

## 3. 使用建议

- **写作流程**：建议直接在对应的 `part_*.tex` 文件中修改内容，而不要改动文件结构。
- **图片替换**：将 `img/` 文件夹下的占位图（如 `cat.jpeg`）替换为你自己的图表。
- **编译方式**：通常需要使用 `XeLaTeX` 编译两次以正确生成目录和交叉引用。
- **代码粘贴**：直接将代码复制到Appendix中的 `matlab` 或 `python` 环境中即可获得美观的排版。
