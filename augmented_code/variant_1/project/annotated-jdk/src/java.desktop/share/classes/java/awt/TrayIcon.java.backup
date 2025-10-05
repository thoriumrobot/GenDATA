/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.*;
import java.awt.peer.TrayIconPeer;
import sun.awt.AppContext;
import sun.awt.SunToolkit;
import sun.awt.AWTAccessor;
import sun.awt.HeadlessToolkit;
import java.util.EventObject;
import java.security.AccessControlContext;
import java.security.AccessController;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class TrayIcon {

    @SuppressWarnings("removal")
    final AccessControlContext getAccessControlContext();

    public TrayIcon(Image image) {
    }

    public TrayIcon(Image image, String tooltip) {
    }

    public TrayIcon(Image image, String tooltip, PopupMenu popup) {
    }

    public void setImage(Image image);

    public Image getImage();

    public void setPopupMenu(PopupMenu popup);

    public PopupMenu getPopupMenu();

    public void setToolTip(String tooltip);

    public String getToolTip();

    public void setImageAutoSize(boolean autosize);

    public boolean isImageAutoSize();

    public synchronized void addMouseListener(MouseListener listener);

    public synchronized void removeMouseListener(MouseListener listener);

    public synchronized MouseListener[] getMouseListeners();

    public synchronized void addMouseMotionListener(MouseMotionListener listener);

    public synchronized void removeMouseMotionListener(MouseMotionListener listener);

    public synchronized MouseMotionListener[] getMouseMotionListeners();

    public String getActionCommand();

    public void setActionCommand(String command);

    public synchronized void addActionListener(ActionListener listener);

    public synchronized void removeActionListener(ActionListener listener);

    public synchronized ActionListener[] getActionListeners();

    public enum MessageType {

        ERROR, WARNING, INFO, NONE
    }

    public void displayMessage(String caption, String text, MessageType messageType);

    public Dimension getSize();

    void addNotify() throws AWTException;

    void removeNotify();

    void setID(int id);

    int getID();

    void dispatchEvent(AWTEvent e);

    void processEvent(AWTEvent e);

    void processMouseEvent(MouseEvent e);

    void processMouseMotionEvent(MouseEvent e);

    void processActionEvent(ActionEvent e);
}
