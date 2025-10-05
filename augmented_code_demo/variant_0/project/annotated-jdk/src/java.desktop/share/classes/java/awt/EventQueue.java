/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.guieffect.qual.UI;
    @Positive
import org.checkerframework.checker.guieffect.qual.UIType;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.awt.event.*;
    @Positive
import java.awt.peer.ComponentPeer;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.EmptyStackException;
    @Positive
import sun.awt.*;
    @Positive
import sun.awt.dnd.SunDropTargetEvent;
    @Positive
import sun.util.logging.PlatformLogger;
    @Positive
import java.util.concurrent.locks.Condition;
    @Positive
import java.util.concurrent.locks.Lock;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.security.AccessControlContext;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.access.JavaSecurityAccess;

    @Positive
@UIType
    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@SuppressWarnings("removal")
    @Positive
@UsesObjectEquals
    @Positive
public class EventQueue {

    @Positive
    public EventQueue() {
    @Positive
    }

    @Positive
    public void postEvent(AWTEvent theEvent);

    @Positive
    public AWTEvent getNextEvent() throws InterruptedException;

    @Positive
    AWTEvent getNextEventPrivate() throws InterruptedException;

    @Positive
    AWTEvent getNextEvent(int id) throws InterruptedException;

    @Positive
    public AWTEvent peekEvent();

    @Positive
    public AWTEvent peekEvent(int id);

    @Positive
    protected void dispatchEvent(final AWTEvent event);

    @Positive
    public static long getMostRecentEventTime();

    @Positive
    long getMostRecentEventTimeEx();

    @Positive
    public static AWTEvent getCurrentEvent();

    @Positive
    public void push(EventQueue newEventQueue);

    @Positive
    protected void pop() throws EmptyStackException;

    @Positive
    public SecondaryLoop createSecondaryLoop();

    @Positive
    private class FwSecondaryLoopWrapper implements SecondaryLoop {

    @Positive
        public FwSecondaryLoopWrapper(SecondaryLoop loop, EventFilter filter) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public boolean enter();

    @Positive
        @Override
    @Positive
        public boolean exit();
    @Positive
    }

    @Positive
    SecondaryLoop createSecondaryLoop(Conditional cond, EventFilter filter, long interval);

    @Positive
    public static boolean isDispatchThread();

    @Positive
    final boolean isDispatchThreadImpl();

    @Positive
    @SuppressWarnings({ "deprecation", "removal" })
    @Positive
    final void initDispatchThread();

    @Positive
    final void detachDispatchThread(EventDispatchThread edt);

    @Positive
    final EventDispatchThread getDispatchThread();

    @Positive
    final void removeSourceEvents(Object source, boolean removeAllEvents);

    @Positive
    synchronized long getMostRecentKeyEventTime();

    @Positive
    static void setCurrentEventAndMostRecentTime(AWTEvent e);

    @Positive
    @SafeEffect
    @Positive
    public static void invokeLater(@UI Runnable runnable);

    @Positive
    public static void invokeAndWait(Runnable runnable) throws InterruptedException, InvocationTargetException;

    @Positive
    static void invokeAndWait(Object source, Runnable runnable) throws InterruptedException, InvocationTargetException;
    @Positive
}

    @Positive
class Queue {
    @Positive
}

// CFWR semantic augmentation - variant 0
