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
package java.awt.dnd;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.GraphicsEnvironment;
import java.awt.HeadlessException;
import java.awt.Insets;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Toolkit;
import java.awt.datatransfer.FlavorMap;
import java.awt.datatransfer.SystemFlavorMap;
import java.awt.dnd.peer.DropTargetPeer;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.peer.ComponentPeer;
import java.awt.peer.LightweightPeer;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.TooManyListenersException;
import javax.swing.Timer;
import sun.awt.AWTAccessor;
import sun.awt.AWTAccessor.ComponentAccessor;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class DropTarget implements DropTargetListener, Serializable {

    public DropTarget(Component c, int ops, DropTargetListener dtl, boolean act, FlavorMap fm) throws HeadlessException {
    }

    public DropTarget(Component c, int ops, DropTargetListener dtl, boolean act) throws HeadlessException {
    }

    public DropTarget() throws HeadlessException {
    }

    public DropTarget(Component c, DropTargetListener dtl) throws HeadlessException {
    }

    public DropTarget(Component c, int ops, DropTargetListener dtl) throws HeadlessException {
    }

    public synchronized void setComponent(Component c);

    public synchronized Component getComponent();

    public void setDefaultActions(int ops);

    void doSetDefaultActions(int ops);

    public int getDefaultActions();

    public synchronized void setActive(boolean isActive);

    public boolean isActive();

    public synchronized void addDropTargetListener(DropTargetListener dtl) throws TooManyListenersException;

    public synchronized void removeDropTargetListener(DropTargetListener dtl);

    public synchronized void dragEnter(DropTargetDragEvent dtde);

    public synchronized void dragOver(DropTargetDragEvent dtde);

    public synchronized void dropActionChanged(DropTargetDragEvent dtde);

    public synchronized void dragExit(DropTargetEvent dte);

    public synchronized void drop(DropTargetDropEvent dtde);

    public FlavorMap getFlavorMap();

    public void setFlavorMap(FlavorMap fm);

    public void addNotify();

    public void removeNotify();

    public DropTargetContext getDropTargetContext();

    protected DropTargetContext createDropTargetContext();

    protected static class DropTargetAutoScroller implements ActionListener {

        protected DropTargetAutoScroller(Component c, Point p) {
        }

        protected synchronized void updateLocation(Point newLocn);

        protected void stop();

        public synchronized void actionPerformed(ActionEvent e);
    }

    protected DropTargetAutoScroller createDropTargetAutoScroller(Component c, Point p);

    protected void initializeAutoscrolling(Point p);

    protected void updateAutoscroll(Point dragCursorLocn);

    protected void clearAutoscroll();
}
