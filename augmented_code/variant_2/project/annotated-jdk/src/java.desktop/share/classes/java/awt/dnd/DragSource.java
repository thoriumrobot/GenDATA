/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.awt.Cursor;
    @Positive
import java.awt.GraphicsEnvironment;
    @Positive
import java.awt.HeadlessException;
    @Positive
import java.awt.Image;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Toolkit;
    @Positive
import java.awt.datatransfer.FlavorMap;
    @Positive
import java.awt.datatransfer.SystemFlavorMap;
    @Positive
import java.awt.datatransfer.Transferable;
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
import java.security.AccessController;
    @Positive
import java.util.EventListener;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.AWTAccessor.DragSourceContextAccessor;
    @Positive
import sun.awt.dnd.SunDragSourceContextPeer;
    @Positive
import sun.security.action.GetIntegerAction;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class DragSource implements Serializable {

    @Positive
    public static final Cursor DefaultCopyDrop;

    @Positive
    public static final Cursor DefaultMoveDrop;

    @Positive
    public static final Cursor DefaultLinkDrop;

    @Positive
    public static final Cursor DefaultCopyNoDrop;

    @Positive
    public static final Cursor DefaultMoveNoDrop;

    @Positive
    public static final Cursor DefaultLinkNoDrop;

    @Positive
    public static DragSource getDefaultDragSource();

    @Positive
    public static boolean isDragImageSupported();

    @Positive
    public DragSource() throws HeadlessException {
    @Positive
    }

    @Positive
    public void startDrag(DragGestureEvent trigger, Cursor dragCursor, Image dragImage, Point imageOffset, Transferable transferable, DragSourceListener dsl, FlavorMap flavorMap) throws InvalidDnDOperationException;

    @Positive
    public void startDrag(DragGestureEvent trigger, Cursor dragCursor, Transferable transferable, DragSourceListener dsl, FlavorMap flavorMap) throws InvalidDnDOperationException;

    @Positive
    public void startDrag(DragGestureEvent trigger, Cursor dragCursor, Image dragImage, Point dragOffset, Transferable transferable, DragSourceListener dsl) throws InvalidDnDOperationException;

    @Positive
    public void startDrag(DragGestureEvent trigger, Cursor dragCursor, Transferable transferable, DragSourceListener dsl) throws InvalidDnDOperationException;

    @Positive
    protected DragSourceContext createDragSourceContext(DragGestureEvent dgl, Cursor dragCursor, Image dragImage, Point imageOffset, Transferable t, DragSourceListener dsl);

    @Positive
    public FlavorMap getFlavorMap();

    @Positive
    public <T extends DragGestureRecognizer> T createDragGestureRecognizer(Class<T> recognizerAbstractClass, Component c, int actions, DragGestureListener dgl);

    @Positive
    public DragGestureRecognizer createDefaultDragGestureRecognizer(Component c, int actions, DragGestureListener dgl);

    @Positive
    public void addDragSourceListener(DragSourceListener dsl);

    @Positive
    public void removeDragSourceListener(DragSourceListener dsl);

    @Positive
    public DragSourceListener[] getDragSourceListeners();

    @Positive
    public void addDragSourceMotionListener(DragSourceMotionListener dsml);

    @Positive
    public void removeDragSourceMotionListener(DragSourceMotionListener dsml);

    @Positive
    public DragSourceMotionListener[] getDragSourceMotionListeners();

    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    void processDragEnter(DragSourceDragEvent dsde);

    @Positive
    void processDragOver(DragSourceDragEvent dsde);

    @Positive
    void processDropActionChanged(DragSourceDragEvent dsde);

    @Positive
    void processDragExit(DragSourceEvent dse);

    @Positive
    void processDragDropEnd(DragSourceDropEvent dsde);

    @Positive
    void processDragMouseMoved(DragSourceDragEvent dsde);

    @Positive
    public static int getDragThreshold();
    @Positive
}
