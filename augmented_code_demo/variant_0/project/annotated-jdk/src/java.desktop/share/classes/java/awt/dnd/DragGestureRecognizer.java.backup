/*
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.Point;
import java.awt.event.InputEvent;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.TooManyListenersException;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class DragGestureRecognizer implements Serializable {

    protected DragGestureRecognizer(DragSource ds, Component c, int sa, DragGestureListener dgl) {
    }

    protected DragGestureRecognizer(DragSource ds, Component c, int sa) {
    }

    protected DragGestureRecognizer(DragSource ds, Component c) {
    }

    protected DragGestureRecognizer(DragSource ds) {
    }

    protected abstract void registerListeners();

    protected abstract void unregisterListeners();

    public DragSource getDragSource();

    public synchronized Component getComponent();

    public synchronized void setComponent(Component c);

    public synchronized int getSourceActions();

    public synchronized void setSourceActions(int actions);

    public InputEvent getTriggerEvent();

    public void resetRecognizer();

    public synchronized void addDragGestureListener(DragGestureListener dgl) throws TooManyListenersException;

    public synchronized void removeDragGestureListener(DragGestureListener dgl);

    protected synchronized void fireDragGestureRecognized(int dragAction, Point p);

    protected synchronized void appendEvent(InputEvent awtie);

    protected DragSource dragSource;

    protected Component component;

    protected transient DragGestureListener dragGestureListener;

    protected int sourceActions;

    protected ArrayList<InputEvent> events;
}
