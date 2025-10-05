/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package javax.swing.text;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import java.io.*;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.text.*;
    @Positive
import javax.swing.Action;
    @Positive
import javax.swing.KeyStroke;
    @Positive
import javax.swing.SwingConstants;
    @Positive
import javax.swing.UIManager;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("serial")
    @Positive
public class DefaultEditorKit extends EditorKit {

    @Positive
    public DefaultEditorKit() {
    @Positive
    }

    @Positive
    public String getContentType();

    @Positive
    public ViewFactory getViewFactory();

    @Positive
    public Action[] getActions();

    @Positive
    public Caret createCaret();

    @Positive
    public Document createDefaultDocument();

    @Positive
    public void read(InputStream in, Document doc, int pos) throws IOException, BadLocationException;

    @Positive
    public void write(OutputStream out, Document doc, int pos, int len) throws IOException, BadLocationException;

    @Positive
    MutableAttributeSet getInputAttributes();

    @Positive
    public void read(Reader in, Document doc, int pos) throws IOException, BadLocationException;

    @Positive
    public void write(Writer out, Document doc, int pos, int len) throws IOException, BadLocationException;

    @Positive
    @Interned
    @Positive
    public static final String EndOfLineStringProperty;

    @Positive
    @Interned
    @Positive
    public static final String insertContentAction;

    @Positive
    @Interned
    @Positive
    public static final String insertBreakAction;

    @Positive
    @Interned
    @Positive
    public static final String insertTabAction;

    @Positive
    @Interned
    @Positive
    public static final String deletePrevCharAction;

    @Positive
    @Interned
    @Positive
    public static final String deleteNextCharAction;

    @Positive
    @Interned
    @Positive
    public static final String deleteNextWordAction;

    @Positive
    @Interned
    @Positive
    public static final String deletePrevWordAction;

    @Positive
    @Interned
    @Positive
    public static final String readOnlyAction;

    @Positive
    @Interned
    @Positive
    public static final String writableAction;

    @Positive
    @Interned
    @Positive
    public static final String cutAction;

    @Positive
    @Interned
    @Positive
    public static final String copyAction;

    @Positive
    @Interned
    @Positive
    public static final String pasteAction;

    @Positive
    @Interned
    @Positive
    public static final String beepAction;

    @Positive
    @Interned
    @Positive
    public static final String pageUpAction;

    @Positive
    @Interned
    @Positive
    public static final String pageDownAction;

    @Positive
    @Interned
    @Positive
    public static final String forwardAction;

    @Positive
    @Interned
    @Positive
    public static final String backwardAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionForwardAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionBackwardAction;

    @Positive
    @Interned
    @Positive
    public static final String upAction;

    @Positive
    @Interned
    @Positive
    public static final String downAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionUpAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionDownAction;

    @Positive
    @Interned
    @Positive
    public static final String beginWordAction;

    @Positive
    @Interned
    @Positive
    public static final String endWordAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionBeginWordAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionEndWordAction;

    @Positive
    @Interned
    @Positive
    public static final String previousWordAction;

    @Positive
    @Interned
    @Positive
    public static final String nextWordAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionPreviousWordAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionNextWordAction;

    @Positive
    @Interned
    @Positive
    public static final String beginLineAction;

    @Positive
    @Interned
    @Positive
    public static final String endLineAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionBeginLineAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionEndLineAction;

    @Positive
    @Interned
    @Positive
    public static final String beginParagraphAction;

    @Positive
    @Interned
    @Positive
    public static final String endParagraphAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionBeginParagraphAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionEndParagraphAction;

    @Positive
    @Interned
    @Positive
    public static final String beginAction;

    @Positive
    @Interned
    @Positive
    public static final String endAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionBeginAction;

    @Positive
    @Interned
    @Positive
    public static final String selectionEndAction;

    @Positive
    @Interned
    @Positive
    public static final String selectWordAction;

    @Positive
    @Interned
    @Positive
    public static final String selectLineAction;

    @Positive
    @Interned
    @Positive
    public static final String selectParagraphAction;

    @Positive
    @Interned
    @Positive
    public static final String selectAllAction;

    @Positive
    @Interned
    @Positive
    public static final String defaultKeyTypedAction;

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class DefaultKeyTypedAction extends TextAction {

    @Positive
        public DefaultKeyTypedAction() {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class InsertContentAction extends TextAction {

    @Positive
        public InsertContentAction() {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class InsertBreakAction extends TextAction {

    @Positive
        public InsertBreakAction() {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class InsertTabAction extends TextAction {

    @Positive
        public InsertTabAction() {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class DeletePrevCharAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class DeleteNextCharAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class DeleteWordAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class ReadOnlyAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class WritableAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class CutAction extends TextAction {

    @Positive
        public CutAction() {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class CopyAction extends TextAction {

    @Positive
        public CopyAction() {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class PasteAction extends TextAction {

    @Positive
        public PasteAction() {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public static class BeepAction extends TextAction {

    @Positive
        public BeepAction() {
    @Positive
        }

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class VerticalPageAction extends TextAction {

    @Positive
        public VerticalPageAction(String nm, int direction, boolean select) {
    @Positive
        }

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class PageAction extends TextAction {

    @Positive
        public PageAction(String nm, boolean left, boolean select) {
    @Positive
        }

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class DumpModelAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class NextVisualPositionAction extends TextAction {

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class BeginWordAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class EndWordAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class PreviousWordAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class NextWordAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class BeginLineAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class EndLineAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class BeginParagraphAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class EndParagraphAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class BeginAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class EndAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class SelectWordAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class SelectLineAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class SelectParagraphAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class SelectAllAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class UnselectAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class ToggleComponentOrientationAction extends TextAction {

    @Positive
        public void actionPerformed(ActionEvent e);
    @Positive
    }
    @Positive
}
