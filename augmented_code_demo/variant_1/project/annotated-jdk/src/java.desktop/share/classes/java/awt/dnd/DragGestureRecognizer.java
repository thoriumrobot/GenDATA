/*
    @Positive
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.awt.Point;
    @Positive
import java.awt.event.InputEvent;
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
import java.util.ArrayList;
    @Positive
import java.util.TooManyListenersException;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class DragGestureRecognizer implements Serializable {

    @Positive
    protected DragGestureRecognizer(DragSource ds, Component c, int sa, DragGestureListener dgl) {
    @Positive
    }

    @Positive
    protected DragGestureRecognizer(DragSource ds, Component c, int sa) {
    @Positive
    }

    @Positive
    protected DragGestureRecognizer(DragSource ds, Component c) {
    @Positive
    }

    @Positive
    protected DragGestureRecognizer(DragSource ds) {
    @Positive
    }

    @Positive
    protected abstract void registerListeners();

    @Positive
    protected abstract void unregisterListeners();

    @Positive
    public DragSource getDragSource();

    @Positive
    public synchronized Component getComponent();

    @Positive
    public synchronized void setComponent(Component c);

    @Positive
    public synchronized int getSourceActions();

    @Positive
    public synchronized void setSourceActions(int actions);

    @Positive
    public InputEvent getTriggerEvent();

    @Positive
    public void resetRecognizer();

    @Positive
    public synchronized void addDragGestureListener(DragGestureListener dgl) throws TooManyListenersException;

    @Positive
    public synchronized void removeDragGestureListener(DragGestureListener dgl);

    @Positive
    protected synchronized void fireDragGestureRecognized(int dragAction, Point p);

    @Positive
    protected synchronized void appendEvent(InputEvent awtie);

    @Positive
    protected DragSource dragSource;

    @Positive
    protected Component component;

    @Positive
    protected transient DragGestureListener dragGestureListener;

    @Positive
    protected int sourceActions;

    @Positive
    protected ArrayList<InputEvent> events;
    @Positive
}

// CFWR semantic augmentation - variant 1
