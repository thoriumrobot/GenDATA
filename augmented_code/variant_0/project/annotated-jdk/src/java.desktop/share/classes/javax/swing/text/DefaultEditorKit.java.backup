/*
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package javax.swing.text;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import sun.awt.SunToolkit;
import java.io.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.text.*;
import javax.swing.Action;
import javax.swing.KeyStroke;
import javax.swing.SwingConstants;
import javax.swing.UIManager;

@AnnotatedFor({ "interning" })
@SuppressWarnings("serial")
public class DefaultEditorKit extends EditorKit {

    public DefaultEditorKit() {
    }

    public String getContentType();

    public ViewFactory getViewFactory();

    public Action[] getActions();

    public Caret createCaret();

    public Document createDefaultDocument();

    public void read(InputStream in, Document doc, int pos) throws IOException, BadLocationException;

    public void write(OutputStream out, Document doc, int pos, int len) throws IOException, BadLocationException;

    MutableAttributeSet getInputAttributes();

    public void read(Reader in, Document doc, int pos) throws IOException, BadLocationException;

    public void write(Writer out, Document doc, int pos, int len) throws IOException, BadLocationException;

    @Interned
    public static final String EndOfLineStringProperty;

    @Interned
    public static final String insertContentAction;

    @Interned
    public static final String insertBreakAction;

    @Interned
    public static final String insertTabAction;

    @Interned
    public static final String deletePrevCharAction;

    @Interned
    public static final String deleteNextCharAction;

    @Interned
    public static final String deleteNextWordAction;

    @Interned
    public static final String deletePrevWordAction;

    @Interned
    public static final String readOnlyAction;

    @Interned
    public static final String writableAction;

    @Interned
    public static final String cutAction;

    @Interned
    public static final String copyAction;

    @Interned
    public static final String pasteAction;

    @Interned
    public static final String beepAction;

    @Interned
    public static final String pageUpAction;

    @Interned
    public static final String pageDownAction;

    @Interned
    public static final String forwardAction;

    @Interned
    public static final String backwardAction;

    @Interned
    public static final String selectionForwardAction;

    @Interned
    public static final String selectionBackwardAction;

    @Interned
    public static final String upAction;

    @Interned
    public static final String downAction;

    @Interned
    public static final String selectionUpAction;

    @Interned
    public static final String selectionDownAction;

    @Interned
    public static final String beginWordAction;

    @Interned
    public static final String endWordAction;

    @Interned
    public static final String selectionBeginWordAction;

    @Interned
    public static final String selectionEndWordAction;

    @Interned
    public static final String previousWordAction;

    @Interned
    public static final String nextWordAction;

    @Interned
    public static final String selectionPreviousWordAction;

    @Interned
    public static final String selectionNextWordAction;

    @Interned
    public static final String beginLineAction;

    @Interned
    public static final String endLineAction;

    @Interned
    public static final String selectionBeginLineAction;

    @Interned
    public static final String selectionEndLineAction;

    @Interned
    public static final String beginParagraphAction;

    @Interned
    public static final String endParagraphAction;

    @Interned
    public static final String selectionBeginParagraphAction;

    @Interned
    public static final String selectionEndParagraphAction;

    @Interned
    public static final String beginAction;

    @Interned
    public static final String endAction;

    @Interned
    public static final String selectionBeginAction;

    @Interned
    public static final String selectionEndAction;

    @Interned
    public static final String selectWordAction;

    @Interned
    public static final String selectLineAction;

    @Interned
    public static final String selectParagraphAction;

    @Interned
    public static final String selectAllAction;

    @Interned
    public static final String defaultKeyTypedAction;

    @SuppressWarnings("serial")
    public static class DefaultKeyTypedAction extends TextAction {

        public DefaultKeyTypedAction() {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    public static class InsertContentAction extends TextAction {

        public InsertContentAction() {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    public static class InsertBreakAction extends TextAction {

        public InsertBreakAction() {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    public static class InsertTabAction extends TextAction {

        public InsertTabAction() {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class DeletePrevCharAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class DeleteNextCharAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class DeleteWordAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class ReadOnlyAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class WritableAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    public static class CutAction extends TextAction {

        public CutAction() {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    public static class CopyAction extends TextAction {

        public CopyAction() {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    public static class PasteAction extends TextAction {

        public PasteAction() {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    public static class BeepAction extends TextAction {

        public BeepAction() {
        }

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class VerticalPageAction extends TextAction {

        public VerticalPageAction(String nm, int direction, boolean select) {
        }

        @SuppressWarnings("deprecation")
        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class PageAction extends TextAction {

        public PageAction(String nm, boolean left, boolean select) {
        }

        @SuppressWarnings("deprecation")
        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class DumpModelAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class NextVisualPositionAction extends TextAction {

        @SuppressWarnings("deprecation")
        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class BeginWordAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class EndWordAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class PreviousWordAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class NextWordAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class BeginLineAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class EndLineAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class BeginParagraphAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class EndParagraphAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class BeginAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class EndAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class SelectWordAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class SelectLineAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class SelectParagraphAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class SelectAllAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class UnselectAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }

    @SuppressWarnings("serial")
    static class ToggleComponentOrientationAction extends TextAction {

        public void actionPerformed(ActionEvent e);
    }
}
