/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.*;
    @Positive
import java.awt.peer.TrayIconPeer;
    @Positive
import sun.awt.AppContext;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.HeadlessToolkit;
    @Positive
import java.util.EventObject;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class TrayIcon {

    @Positive
    @SuppressWarnings("removal")
    @Positive
    final AccessControlContext getAccessControlContext();

    @Positive
    public TrayIcon(Image image) {
    @Positive
    }

    @Positive
    public TrayIcon(Image image, String tooltip) {
    @Positive
    }

    @Positive
    public TrayIcon(Image image, String tooltip, PopupMenu popup) {
    @Positive
    }

    @Positive
    public void setImage(Image image);

    @Positive
    public Image getImage();

    @Positive
    public void setPopupMenu(PopupMenu popup);

    @Positive
    public PopupMenu getPopupMenu();

    @Positive
    public void setToolTip(String tooltip);

    @Positive
    public String getToolTip();

    @Positive
    public void setImageAutoSize(boolean autosize);

    @Positive
    public boolean isImageAutoSize();

    @Positive
    public synchronized void addMouseListener(MouseListener listener);

    @Positive
    public synchronized void removeMouseListener(MouseListener listener);

    @Positive
    public synchronized MouseListener[] getMouseListeners();

    @Positive
    public synchronized void addMouseMotionListener(MouseMotionListener listener);

    @Positive
    public synchronized void removeMouseMotionListener(MouseMotionListener listener);

    @Positive
    public synchronized MouseMotionListener[] getMouseMotionListeners();

    @Positive
    public String getActionCommand();

    @Positive
    public void setActionCommand(String command);

    @Positive
    public synchronized void addActionListener(ActionListener listener);

    @Positive
    public synchronized void removeActionListener(ActionListener listener);

    @Positive
    public synchronized ActionListener[] getActionListeners();

    @Positive
    public enum MessageType {

    @Positive
        ERROR, WARNING, INFO, NONE
    @Positive
    }

    @Positive
    public void displayMessage(String caption, String text, MessageType messageType);

    @Positive
    public Dimension getSize();

    @Positive
    void addNotify() throws AWTException;

    @Positive
    void removeNotify();

    @Positive
    void setID(int id);

    @Positive
    int getID();

    @Positive
    void dispatchEvent(AWTEvent e);

    @Positive
    void processEvent(AWTEvent e);

    @Positive
    void processMouseEvent(MouseEvent e);

    @Positive
    void processMouseMotionEvent(MouseEvent e);

    @Positive
    void processActionEvent(ActionEvent e);
    @Positive
}
