/*
 * Copyright (c) 1997, 2013, Oracle and/or its affiliates. All rights reserved.
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
import javax.swing.event.*;

@AnnotatedFor({ "interning" })
public interface Document {

    public int getLength();

    public void addDocumentListener(DocumentListener listener);

    public void removeDocumentListener(DocumentListener listener);

    public void addUndoableEditListener(UndoableEditListener listener);

    public void removeUndoableEditListener(UndoableEditListener listener);

    public Object getProperty(Object key);

    public void putProperty(Object key, Object value);

    public void remove(int offs, int len) throws BadLocationException;

    public void insertString(int offset, String str, AttributeSet a) throws BadLocationException;

    public String getText(int offset, int length) throws BadLocationException;

    public void getText(int offset, int length, Segment txt) throws BadLocationException;

    public Position getStartPosition();

    public Position getEndPosition();

    public Position createPosition(int offs) throws BadLocationException;

    public Element[] getRootElements();

    public Element getDefaultRootElement();

    public void render(Runnable r);

    @Interned
    public static final String StreamDescriptionProperty;

    @Interned
    public static final String TitleProperty;
}
