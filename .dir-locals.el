;; TAB indentation matching .clang-format (emacs-cpp docs/MANUAL.md 8, D-062).
((c++-ts-mode
  . ((c-ts-indent-offset . 4)
     (indent-tabs-mode . nil)
     (c-ts-mode-indent-style . bsd)
     (emacs-cpp-indent-namespace-body . nil)
     (emacs-cpp-indent-case-labels . t)
     (dape-command . (gdb-preset :args ["--levels" "5"]))))
 ;; ty infers the return types of unannotated functions (emacs-cpp D-142).
 (python-ts-mode
  . ((emacs-cpp-ty-return-type-inference . t))))
