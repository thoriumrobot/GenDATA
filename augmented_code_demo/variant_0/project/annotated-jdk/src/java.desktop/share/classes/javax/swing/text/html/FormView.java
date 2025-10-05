/*
    @Positive
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.text.html;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.net.*;
    @Positive
import java.io.*;
    @Positive
import java.awt.*;
    @Positive
import java.awt.event.*;
    @Positive
import java.util.*;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.event.*;
    @Positive
import javax.swing.text.*;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class FormView extends ComponentView implements ActionListener {

    @Positive
    @Deprecated
    @Positive
    @Interned
    @Positive
    public static final String SUBMIT;

    @Positive
    @Deprecated
    @Positive
    @Interned
    @Positive
    public static final String RESET;

    @Positive
    public FormView(Element elem) {
    @Positive
    }

    @Positive
    protected Component createComponent();

    @Positive
    public float getMaximumSpan(int axis);

    @Positive
    public void actionPerformed(ActionEvent evt);

    @Positive
    protected void submitData(String data);

    @Positive
    protected class MouseEventListener extends MouseAdapter {

    @Positive
        protected MouseEventListener() {
    @Positive
        }

    @Positive
        public void mouseReleased(MouseEvent evt);
    @Positive
    }

    @Positive
    protected void imageSubmit(String imageData);

    @Positive
    boolean isLastTextOrPasswordField();

    @Positive
    void resetForm();

    @Positive
    private class BrowseFileAction implements ActionListener {

    @Positive
        public void actionPerformed(ActionEvent ae);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
