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
package java.awt.dnd;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.GraphicsEnvironment;
    @Positive
import java.awt.HeadlessException;
    @Positive
import java.awt.Insets;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.datatransfer.FlavorMap;
    @Positive
import java.awt.datatransfer.SystemFlavorMap;
    @Positive
import java.awt.dnd.peer.DropTargetPeer;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.ActionListener;
    @Positive
import java.awt.peer.ComponentPeer;
    @Positive
import java.awt.peer.LightweightPeer;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.TooManyListenersException;
    @Positive
import javax.swing.Timer;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AWTAccessor.ComponentAccessor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class DropTarget implements DropTargetListener, Serializable {

    @Positive
    public DropTarget(Component c, int ops, DropTargetListener dtl, boolean act, FlavorMap fm) throws HeadlessException {
    @Positive
    }

    @Positive
    public DropTarget(Component c, int ops, DropTargetListener dtl, boolean act) throws HeadlessException {
    @Positive
    }

    @Positive
    public DropTarget() throws HeadlessException {
    @Positive
    }

    @Positive
    public DropTarget(Component c, DropTargetListener dtl) throws HeadlessException {
    @Positive
    }

    @Positive
    public DropTarget(Component c, int ops, DropTargetListener dtl) throws HeadlessException {
    @Positive
    }

    @Positive
    public synchronized void setComponent(Component c);

    @Positive
    public synchronized Component getComponent();

    @Positive
    public void setDefaultActions(int ops);

    @Positive
    void doSetDefaultActions(int ops);

    @Positive
    public int getDefaultActions();

    @Positive
    public synchronized void setActive(boolean isActive);

    @Positive
    public boolean isActive();

    @Positive
    public synchronized void addDropTargetListener(DropTargetListener dtl) throws TooManyListenersException;

    @Positive
    public synchronized void removeDropTargetListener(DropTargetListener dtl);

    @Positive
    public synchronized void dragEnter(DropTargetDragEvent dtde);

    @Positive
    public synchronized void dragOver(DropTargetDragEvent dtde);

    @Positive
    public synchronized void dropActionChanged(DropTargetDragEvent dtde);

    @Positive
    public synchronized void dragExit(DropTargetEvent dte);

    @Positive
    public synchronized void drop(DropTargetDropEvent dtde);

    @Positive
    public FlavorMap getFlavorMap();

    @Positive
    public void setFlavorMap(FlavorMap fm);

    @Positive
    public void addNotify();

    @Positive
    public void removeNotify();

    @Positive
    public DropTargetContext getDropTargetContext();

    @Positive
    protected DropTargetContext createDropTargetContext();

    @Positive
    protected static class DropTargetAutoScroller implements ActionListener {

    @Positive
        protected DropTargetAutoScroller(Component c, Point p) {
    @Positive
        }

    @Positive
        protected synchronized void updateLocation(Point newLocn);

    @Positive
        protected void stop();

    @Positive
        public synchronized void actionPerformed(ActionEvent e);
    @Positive
    }

    @Positive
    protected DropTargetAutoScroller createDropTargetAutoScroller(Component c, Point p);

    @Positive
    protected void initializeAutoscrolling(Point p);

    @Positive
    protected void updateAutoscroll(Point dragCursorLocn);

    @Positive
    protected void clearAutoscroll();
    @Positive
}

// CFWR semantic augmentation - variant 1
