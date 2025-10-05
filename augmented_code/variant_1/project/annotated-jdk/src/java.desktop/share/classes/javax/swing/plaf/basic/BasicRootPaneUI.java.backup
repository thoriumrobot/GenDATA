/*
 * Copyright (c) 1999, 2016, Oracle and/or its affiliates. All rights reserved.
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
package javax.swing.plaf.basic;

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.ActionEvent;
import java.awt.KeyboardFocusManager;
import java.awt.Component;
import java.awt.Point;
import java.awt.Rectangle;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import javax.swing.*;
import javax.swing.plaf.*;
import sun.swing.DefaultLookup;
import sun.swing.UIAction;

@AnnotatedFor({ "interning" })
public class BasicRootPaneUI extends RootPaneUI implements PropertyChangeListener {

    public BasicRootPaneUI() {
    }

    public static ComponentUI createUI(JComponent c);

    public void installUI(JComponent c);

    public void uninstallUI(JComponent c);

    protected void installDefaults(JRootPane c);

    protected void installComponents(JRootPane root);

    protected void installListeners(JRootPane root);

    protected void installKeyboardActions(JRootPane root);

    protected void uninstallDefaults(JRootPane root);

    protected void uninstallComponents(JRootPane root);

    protected void uninstallListeners(JRootPane root);

    protected void uninstallKeyboardActions(JRootPane root);

    InputMap getInputMap(int condition, JComponent c);

    ComponentInputMap createInputMap(int condition, JComponent c);

    static void loadActionMap(LazyActionMap map);

    void updateDefaultButtonBindings(JRootPane root);

    public void propertyChange(PropertyChangeEvent e);

    static class Actions extends UIAction {

        @Interned
        public static final String PRESS;

        @Interned
        public static final String RELEASE;

        @Interned
        public static final String POST_POPUP;

        public void actionPerformed(ActionEvent evt);

        @Override
        public boolean accept(Object sender);
    }

    @SuppressWarnings("serial")
    private static class RootPaneInputMap extends ComponentInputMapUIResource {

        public RootPaneInputMap(JComponent c) {
        }
    }
}
