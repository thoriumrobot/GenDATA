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
import java.awt.AWTError;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Image;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.dnd.peer.DragSourceContextPeer;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.TooManyListenersException;
import sun.awt.AWTAccessor;
import sun.awt.ComponentFactory;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class DragSourceContext implements DragSourceListener, DragSourceMotionListener, Serializable {

    protected static final int DEFAULT;

    protected static final int ENTER;

    protected static final int OVER;

    protected static final int CHANGED;

    public DragSourceContext(DragGestureEvent trigger, Cursor dragCursor, Image dragImage, Point offset, Transferable t, DragSourceListener dsl) {
    }

    public DragSource getDragSource();

    public Component getComponent();

    public DragGestureEvent getTrigger();

    public int getSourceActions();

    public synchronized void setCursor(Cursor c);

    public Cursor getCursor();

    public synchronized void addDragSourceListener(DragSourceListener dsl) throws TooManyListenersException;

    public synchronized void removeDragSourceListener(DragSourceListener dsl);

    public void transferablesFlavorsChanged();

    public void dragEnter(DragSourceDragEvent dsde);

    public void dragOver(DragSourceDragEvent dsde);

    public void dragExit(DragSourceEvent dse);

    public void dropActionChanged(DragSourceDragEvent dsde);

    public void dragDropEnd(DragSourceDropEvent dsde);

    public void dragMouseMoved(DragSourceDragEvent dsde);

    public Transferable getTransferable();

    @SuppressWarnings("fallthrough")
    protected synchronized void updateCurrentCursor(int sourceAct, int targetAct, int status);
}
