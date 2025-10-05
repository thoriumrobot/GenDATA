/*
    @Positive
 * Copyright (c) 1999, 2016, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.basic;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.KeyboardFocusManager;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.beans.PropertyChangeEvent;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import javax.swing.*;
    @Positive
import javax.swing.plaf.*;
    @Positive
import sun.swing.DefaultLookup;
    @Positive
import sun.swing.UIAction;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class BasicRootPaneUI extends RootPaneUI implements PropertyChangeListener {

    @Positive
    public BasicRootPaneUI() {
    @Positive
    }

    @Positive
    public static ComponentUI createUI(JComponent c);

    @Positive
    public void installUI(JComponent c);

    @Positive
    public void uninstallUI(JComponent c);

    @Positive
    protected void installDefaults(JRootPane c);

    @Positive
    protected void installComponents(JRootPane root);

    @Positive
    protected void installListeners(JRootPane root);

    @Positive
    protected void installKeyboardActions(JRootPane root);

    @Positive
    protected void uninstallDefaults(JRootPane root);

    @Positive
    protected void uninstallComponents(JRootPane root);

    @Positive
    protected void uninstallListeners(JRootPane root);

    @Positive
    protected void uninstallKeyboardActions(JRootPane root);

    @Positive
    InputMap getInputMap(int condition, JComponent c);

    @Positive
    ComponentInputMap createInputMap(int condition, JComponent c);

    @Positive
    static void loadActionMap(LazyActionMap map);

    @Positive
    void updateDefaultButtonBindings(JRootPane root);

    @Positive
    public void propertyChange(PropertyChangeEvent e);

    @Positive
    static class Actions extends UIAction {

    @Positive
        @Interned
    @Positive
        public static final String PRESS;

    @Positive
        @Interned
    @Positive
        public static final String RELEASE;

    @Positive
        @Interned
    @Positive
        public static final String POST_POPUP;

    @Positive
        public void actionPerformed(ActionEvent evt);

    @Positive
        @Override
    @Positive
        public boolean accept(Object sender);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    private static class RootPaneInputMap extends ComponentInputMapUIResource {

    @Positive
        public RootPaneInputMap(JComponent c) {
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
