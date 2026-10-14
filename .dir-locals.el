;; TAB indentation matching .clang-format (emacs-cpp docs/MANUAL.md 8, D-062).
((c++-ts-mode
  . ((c-ts-indent-offset . 4)
     (indent-tabs-mode . nil)
     (c-ts-mode-indent-style . bsd)
     (emacs-cpp-indent-namespace-body . nil)
     (emacs-cpp-indent-case-labels . t)
     (dape-command . (gdb-preset :args ["--levels" "5"])))))
