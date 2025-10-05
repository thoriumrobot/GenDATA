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
import java.awt.Cursor;
import java.awt.GraphicsEnvironment;
import java.awt.HeadlessException;
import java.awt.Image;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.datatransfer.FlavorMap;
import java.awt.datatransfer.SystemFlavorMap;
import java.awt.datatransfer.Transferable;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.security.AccessController;
import java.util.EventListener;
import sun.awt.AWTAccessor;
import sun.awt.AWTAccessor.DragSourceContextAccessor;
import sun.awt.dnd.SunDragSourceContextPeer;
import sun.security.action.GetIntegerAction;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class DragSource implements Serializable {

    public static final Cursor DefaultCopyDrop;

    public static final Cursor DefaultMoveDrop;

    public static final Cursor DefaultLinkDrop;

    public static final Cursor DefaultCopyNoDrop;

    public static final Cursor DefaultMoveNoDrop;

    public static final Cursor DefaultLinkNoDrop;

    public static DragSource getDefaultDragSource();

    public static boolean isDragImageSupported();

    public DragSource() throws HeadlessException {
    }

    public void startDrag(DragGestureEvent trigger, Cursor dragCursor, Image dragImage, Point imageOffset, Transferable transferable, DragSourceListener dsl, FlavorMap flavorMap) throws InvalidDnDOperationException;

    public void startDrag(DragGestureEvent trigger, Cursor dragCursor, Transferable transferable, DragSourceListener dsl, FlavorMap flavorMap) throws InvalidDnDOperationException;

    public void startDrag(DragGestureEvent trigger, Cursor dragCursor, Image dragImage, Point dragOffset, Transferable transferable, DragSourceListener dsl) throws InvalidDnDOperationException;

    public void startDrag(DragGestureEvent trigger, Cursor dragCursor, Transferable transferable, DragSourceListener dsl) throws InvalidDnDOperationException;

    protected DragSourceContext createDragSourceContext(DragGestureEvent dgl, Cursor dragCursor, Image dragImage, Point imageOffset, Transferable t, DragSourceListener dsl);

    public FlavorMap getFlavorMap();

    public <T extends DragGestureRecognizer> T createDragGestureRecognizer(Class<T> recognizerAbstractClass, Component c, int actions, DragGestureListener dgl);

    public DragGestureRecognizer createDefaultDragGestureRecognizer(Component c, int actions, DragGestureListener dgl);

    public void addDragSourceListener(DragSourceListener dsl);

    public void removeDragSourceListener(DragSourceListener dsl);

    public DragSourceListener[] getDragSourceListeners();

    public void addDragSourceMotionListener(DragSourceMotionListener dsml);

    public void removeDragSourceMotionListener(DragSourceMotionListener dsml);

    public DragSourceMotionListener[] getDragSourceMotionListeners();

    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    void processDragEnter(DragSourceDragEvent dsde);

    void processDragOver(DragSourceDragEvent dsde);

    void processDropActionChanged(DragSourceDragEvent dsde);

    void processDragExit(DragSourceEvent dse);

    void processDragDropEnd(DragSourceDropEvent dsde);

    void processDragMouseMoved(DragSourceDragEvent dsde);

    public static int getDragThreshold();
}
