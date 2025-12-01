
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{dtuarticle}[Article class for DTU]
\providecommand{\Logo}{dtu-text}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{dtu}
\ProcessOptions\relax
\LoadClass{article}
\RequirePackage[a4paper]{geometry}

%\RequirePackage{arev}
%\renewcommand*\familydefault{\sfdefault} %% Only if the base font of the document is to be sans serif
\RequirePackage[T1]{fontenc}
\RequirePackage{hyperref}
\RequirePackage{titling}
\RequirePackage{multicol}
\RequirePackage{graphicx,xcolor}
\definecolor{grijs}{RGB}{218,218,218}
\definecolor{dtured}{RGB}{153,0,0}
\RequirePackage{lastpage}
\RequirePackage{tikz}
\usetikzlibrary{arrows}
\usetikzlibrary{decorations.pathmorphing}
\usetikzlibrary{backgrounds}
\usetikzlibrary{fit}
\usetikzlibrary{shadows}
\usetikzlibrary{positioning}
\usetikzlibrary{calc}
\makeatletter
\newcommand\subtitle[1]{\renewcommand\@subtitle{#1}}
\newcommand\@subtitle{}
\newcommand{\thesubtitle}{\@subtitle{}}
\newcommand\address[1]{\renewcommand\@address{#1}}
\newcommand\@address{}
\newcommand{\theaddress}{\@address{}}
\newcommand\course[1]{\renewcommand\@course{#1}}
\newcommand\@course{}
\newcommand{\thecourse}{\@course{}}
\newcommand\location[1]{\renewcommand\@location{#1}}
\newcommand\@location{}
\newcommand{\thelocation}{\@location{}}
\makeatother
\setlength{\topmargin}{-1in}
\addtolength{\topmargin}{1.5cm}
\setlength{\oddsidemargin}{-1in}
\addtolength{\oddsidemargin}{3cm}
\setlength{\textwidth}{16cm}
\setlength{\textheight}{22.5cm}
\setlength{\footskip}{\dimexpr\footskip+0.4cm}
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\setlength{\headheight}{1.5cm}
\lhead{}
\chead{}
\rhead{\begin{tikzpicture}[inner sep=0mm]
		\node[anchor=west] at ( 0.5,0.6) {\includegraphics[height=1.5cm]{dtu-logo.pdf}};
\end{tikzpicture}}
\lfoot{\begin{tikzpicture}[overlay, inner sep=0mm]
		\fill[grijs]  (0,1.1) rectangle ( 16,1.12);
	\end{tikzpicture}\renewcommand{\arraystretch}{0.8}
	\begin{tabular}[b]{@{}l@{}}
		\footnotesize{\textbf{Technical University of Denmark} - \thecourse}\\
		\footnotesize{\thetitle} - \footnotesize{\textit{\thesubtitle}}\\
		\footnotesize{\theauthor}
\end{tabular}}
\cfoot{}
\rfoot{\footnotesize{\thepage/\pageref{LastPage}}}

\renewcommand{\maketitle}{
	\begin{tikzpicture}[remember picture,overlay]
		\node at (current page.south west)
		{
			\begin{tikzpicture}[remember picture, overlay,inner sep=0mm]
				\node[opacity=0.35,inner sep=0pt] at ($(current page.north west) + (4cm, -3cm)$){\includegraphics[width=\paperwidth]{math-pattern}};
				\fill[grijs] (17.58,26.87) rectangle (17.62,28.07);
				\node[anchor=north west] at ( 3.0,28.2) {\includegraphics[height=1.5cm]{\Logo}};
				\node[anchor=north east] at ( 19,28.2) {\includegraphics[height=1.5cm]{dtu-logo.pdf}};
				\node[anchor=east] at (17.1,27.47) {
					\footnotesize\begin{tabular}{@{}r@{}}\textbf{\thecourse}\\ \theaddress\end{tabular}
				};
				\node[anchor=south east] at (19,24) {
					\begin{tabular}{@{}r@{}}
						\large\bf\thetitle\\\textit{\thesubtitle}\\[.5cm]
						\large\theauthor\vphantom{Ky}
				\end{tabular}};
				\node[anchor=south west] at (3,24) {
					\begin{tabular}{@{}l@{}}
						\large \thedate\vphantom{Ky}\\
				\end{tabular}};
				\fill[grijs]  (3,23.66) rectangle ( 19,23.7);
			\end{tikzpicture}
		};
	\end{tikzpicture}
	\vspace*{1.5cm}\\
	{}
	\thispagestyle{empty}
}