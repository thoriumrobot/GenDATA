/*
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.text.html;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.net.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.text.*;

@AnnotatedFor({ "interning" })
public class FormView extends ComponentView implements ActionListener {

    @Deprecated
    @Interned
    public static final String SUBMIT;

    @Deprecated
    @Interned
    public static final String RESET;

    public FormView(Element elem) {
    }

    protected Component createComponent();

    public float getMaximumSpan(int axis);

    public void actionPerformed(ActionEvent evt);

    protected void submitData(String data);

    protected class MouseEventListener extends MouseAdapter {

        protected MouseEventListener() {
        }

        public void mouseReleased(MouseEvent evt);
    }

    protected void imageSubmit(String imageData);

    boolean isLastTextOrPasswordField();

    void resetForm();

    private class BrowseFileAction implements ActionListener {

        public void actionPerformed(ActionEvent ae);
    }
}
