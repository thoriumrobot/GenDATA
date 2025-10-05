/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.awt.im;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.AWTEvent;
    @Positive
import java.awt.AWTKeyStroke;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.EventQueue;
    @Positive
import java.awt.Frame;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.Window;
    @Positive
import java.awt.event.ComponentEvent;
    @Positive
import java.awt.event.ComponentListener;
    @Positive
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.InputEvent;
    @Positive
import java.awt.event.InputMethodEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.WindowEvent;
    @Positive
import java.awt.event.WindowListener;
    @Positive
import java.awt.im.InputMethodRequests;
    @Positive
import java.awt.im.spi.InputMethod;
    @Positive
import java.lang.Character.Subset;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.text.MessageFormat;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Locale;
    @Positive
import java.util.prefs.BackingStoreException;
    @Positive
import java.util.prefs.Preferences;
    @Positive
import sun.util.logging.PlatformLogger;
    @Positive
import sun.awt.SunToolkit;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class InputContext extends java.awt.im.InputContext implements ComponentListener, WindowListener {

    @Positive
    protected InputContext() {
    @Positive
    }

    @Positive
    public synchronized boolean selectInputMethod(Locale locale);

    @Positive
    public Locale getLocale();

    @Positive
    public void setCharacterSubsets(Subset[] subsets);

    @Positive
    public synchronized void reconvert();

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    public void dispatchEvent(AWTEvent event);

    @Positive
    static Window getComponentWindow(Component component);

    @Positive
    synchronized void changeInputMethod(InputMethodLocator newLocator);

    @Positive
    Component getClientComponent();

    @Positive
    public synchronized void removeNotify(Component component);

    @Positive
    public synchronized void dispose();

    @Positive
    public synchronized Object getInputMethodControlObject();

    @Positive
    public void setCompositionEnabled(boolean enable);

    @Positive
    public boolean isCompositionEnabled();

    @Positive
    public String getInputMethodInfo();

    @Positive
    public void disableNativeIM();

    @Positive
    InputMethodLocator getInputMethodLocator();

    @Positive
    public synchronized void endComposition();

    @Positive
    synchronized void enableClientWindowNotification(InputMethod requester, boolean enable);

    @Positive
    public void componentResized(ComponentEvent e);

    @Positive
    public void componentMoved(ComponentEvent e);

    @Positive
    public void componentShown(ComponentEvent e);

    @Positive
    public void componentHidden(ComponentEvent e);

    @Positive
    public void windowOpened(WindowEvent e);

    @Positive
    public void windowClosing(WindowEvent e);

    @Positive
    public void windowClosed(WindowEvent e);

    @Positive
    public void windowIconified(WindowEvent e);

    @Positive
    public void windowDeiconified(WindowEvent e);

    @Positive
    public void windowActivated(WindowEvent e);

    @Positive
    public void windowDeactivated(WindowEvent e);
    @Positive
}

// CFWR semantic augmentation - variant 0
