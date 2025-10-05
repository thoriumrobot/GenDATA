/*
    @Positive
 * Copyright (c) 1997, 2013, Oracle and/or its affiliates. All rights reserved.
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
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
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
import javax.swing.event.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public interface Document {

    @Positive
    public int getLength();

    @Positive
    public void addDocumentListener(DocumentListener listener);

    @Positive
    public void removeDocumentListener(DocumentListener listener);

    @Positive
    public void addUndoableEditListener(UndoableEditListener listener);

    @Positive
    public void removeUndoableEditListener(UndoableEditListener listener);

    @Positive
    public Object getProperty(Object key);

    @Positive
    public void putProperty(Object key, Object value);

    @Positive
    public void remove(int offs, int len) throws BadLocationException;

    @Positive
    public void insertString(int offset, String str, AttributeSet a) throws BadLocationException;

    @Positive
    public String getText(int offset, int length) throws BadLocationException;

    @Positive
    public void getText(int offset, int length, Segment txt) throws BadLocationException;

    @Positive
    public Position getStartPosition();

    @Positive
    public Position getEndPosition();

    @Positive
    public Position createPosition(int offs) throws BadLocationException;

    @Positive
    public Element[] getRootElements();

    @Positive
    public Element getDefaultRootElement();

    @Positive
    public void render(Runnable r);

    @Positive
    @Interned
    @Positive
    public static final String StreamDescriptionProperty;

    @Positive
    @Interned
    @Positive
    public static final String TitleProperty;
    @Positive
}

// CFWR semantic augmentation - variant 1
