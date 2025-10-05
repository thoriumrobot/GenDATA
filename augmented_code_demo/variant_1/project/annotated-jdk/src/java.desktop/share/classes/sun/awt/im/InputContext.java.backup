/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.awt.im;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.AWTEvent;
import java.awt.AWTKeyStroke;
import java.awt.Component;
import java.awt.EventQueue;
import java.awt.Frame;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.FocusEvent;
import java.awt.event.InputEvent;
import java.awt.event.InputMethodEvent;
import java.awt.event.KeyEvent;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.im.InputMethodRequests;
import java.awt.im.spi.InputMethod;
import java.lang.Character.Subset;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.text.MessageFormat;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Locale;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;
import sun.util.logging.PlatformLogger;
import sun.awt.SunToolkit;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class InputContext extends java.awt.im.InputContext implements ComponentListener, WindowListener {

    protected InputContext() {
    }

    public synchronized boolean selectInputMethod(Locale locale);

    public Locale getLocale();

    public void setCharacterSubsets(Subset[] subsets);

    public synchronized void reconvert();

    @SuppressWarnings("fallthrough")
    public void dispatchEvent(AWTEvent event);

    static Window getComponentWindow(Component component);

    synchronized void changeInputMethod(InputMethodLocator newLocator);

    Component getClientComponent();

    public synchronized void removeNotify(Component component);

    public synchronized void dispose();

    public synchronized Object getInputMethodControlObject();

    public void setCompositionEnabled(boolean enable);

    public boolean isCompositionEnabled();

    public String getInputMethodInfo();

    public void disableNativeIM();

    InputMethodLocator getInputMethodLocator();

    public synchronized void endComposition();

    synchronized void enableClientWindowNotification(InputMethod requester, boolean enable);

    public void componentResized(ComponentEvent e);

    public void componentMoved(ComponentEvent e);

    public void componentShown(ComponentEvent e);

    public void componentHidden(ComponentEvent e);

    public void windowOpened(WindowEvent e);

    public void windowClosing(WindowEvent e);

    public void windowClosed(WindowEvent e);

    public void windowIconified(WindowEvent e);

    public void windowDeiconified(WindowEvent e);

    public void windowActivated(WindowEvent e);

    public void windowDeactivated(WindowEvent e);
}
