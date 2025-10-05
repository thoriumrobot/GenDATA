/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.guieffect.qual.SafeEffect;
import org.checkerframework.checker.guieffect.qual.UI;
import org.checkerframework.checker.guieffect.qual.UIType;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.*;
import java.awt.peer.ComponentPeer;
import java.lang.ref.WeakReference;
import java.lang.reflect.InvocationTargetException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.EmptyStackException;
import sun.awt.*;
import sun.awt.dnd.SunDropTargetEvent;
import sun.util.logging.PlatformLogger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.atomic.AtomicInteger;
import java.security.AccessControlContext;
import jdk.internal.access.SharedSecrets;
import jdk.internal.access.JavaSecurityAccess;

@UIType
@AnnotatedFor({ "interning" })
@SuppressWarnings("removal")
@UsesObjectEquals
public class EventQueue {

    public EventQueue() {
    }

    public void postEvent(AWTEvent theEvent);

    public AWTEvent getNextEvent() throws InterruptedException;

    AWTEvent getNextEventPrivate() throws InterruptedException;

    AWTEvent getNextEvent(int id) throws InterruptedException;

    public AWTEvent peekEvent();

    public AWTEvent peekEvent(int id);

    protected void dispatchEvent(final AWTEvent event);

    public static long getMostRecentEventTime();

    long getMostRecentEventTimeEx();

    public static AWTEvent getCurrentEvent();

    public void push(EventQueue newEventQueue);

    protected void pop() throws EmptyStackException;

    public SecondaryLoop createSecondaryLoop();

    private class FwSecondaryLoopWrapper implements SecondaryLoop {

        public FwSecondaryLoopWrapper(SecondaryLoop loop, EventFilter filter) {
        }

        @Override
        public boolean enter();

        @Override
        public boolean exit();
    }

    SecondaryLoop createSecondaryLoop(Conditional cond, EventFilter filter, long interval);

    public static boolean isDispatchThread();

    final boolean isDispatchThreadImpl();

    @SuppressWarnings({ "deprecation", "removal" })
    final void initDispatchThread();

    final void detachDispatchThread(EventDispatchThread edt);

    final EventDispatchThread getDispatchThread();

    final void removeSourceEvents(Object source, boolean removeAllEvents);

    synchronized long getMostRecentKeyEventTime();

    static void setCurrentEventAndMostRecentTime(AWTEvent e);

    @SafeEffect
    public static void invokeLater(@UI Runnable runnable);

    public static void invokeAndWait(Runnable runnable) throws InterruptedException, InvocationTargetException;

    static void invokeAndWait(Object source, Runnable runnable) throws InterruptedException, InvocationTargetException;
}

class Queue {
}
