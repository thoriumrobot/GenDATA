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
import java.awt.AWTError;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Cursor;
    @Positive
import java.awt.Image;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.datatransfer.DataFlavor;
    @Positive
import java.awt.datatransfer.Transferable;
    @Positive
import java.awt.datatransfer.UnsupportedFlavorException;
    @Positive
import java.awt.dnd.peer.DragSourceContextPeer;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
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
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.ComponentFactory;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class DragSourceContext implements DragSourceListener, DragSourceMotionListener, Serializable {

    @Positive
    protected static final int DEFAULT;

    @Positive
    protected static final int ENTER;

    @Positive
    protected static final int OVER;

    @Positive
    protected static final int CHANGED;

    @Positive
    public DragSourceContext(DragGestureEvent trigger, Cursor dragCursor, Image dragImage, Point offset, Transferable t, DragSourceListener dsl) {
    @Positive
    }

    @Positive
    public DragSource getDragSource();

    @Positive
    public Component getComponent();

    @Positive
    public DragGestureEvent getTrigger();

    @Positive
    public int getSourceActions();

    @Positive
    public synchronized void setCursor(Cursor c);

    @Positive
    public Cursor getCursor();

    @Positive
    public synchronized void addDragSourceListener(DragSourceListener dsl) throws TooManyListenersException;

    @Positive
    public synchronized void removeDragSourceListener(DragSourceListener dsl);

    @Positive
    public void transferablesFlavorsChanged();

    @Positive
    public void dragEnter(DragSourceDragEvent dsde);

    @Positive
    public void dragOver(DragSourceDragEvent dsde);

    @Positive
    public void dragExit(DragSourceEvent dse);

    @Positive
    public void dropActionChanged(DragSourceDragEvent dsde);

    @Positive
    public void dragDropEnd(DragSourceDropEvent dsde);

    @Positive
    public void dragMouseMoved(DragSourceDragEvent dsde);

    @Positive
    public Transferable getTransferable();

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    protected synchronized void updateCurrentCursor(int sourceAct, int targetAct, int status);
    @Positive
}

// CFWR semantic augmentation - variant 0
