/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.io.PrintStream;
import java.util.Arrays;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@UsesObjectEquals
public class ThreadGroup implements Thread.UncaughtExceptionHandler {

    public ThreadGroup(@Nullable String name) {
    }

    public ThreadGroup(ThreadGroup parent, @Nullable String name) {
    }

    @Nullable
    public final String getName();

    @Nullable
    public final ThreadGroup getParent();

    public final int getMaxPriority();

    @Pure
    @Deprecated()
    public final boolean isDaemon(@GuardSatisfied ThreadGroup this);

    @Pure
    @Deprecated()
    public synchronized boolean isDestroyed(@GuardSatisfied ThreadGroup this);

    @Deprecated()
    public final void setDaemon(boolean daemon);

    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @SuppressWarnings("index:array.access.unsafe.high")
    public final void setMaxPriority(int pri);

    public final boolean parentOf(ThreadGroup g);

    @Deprecated()
    public final void checkAccess();

    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @SuppressWarnings("index:array.access.unsafe.high")
    @NonNegative
    public int activeCount();

    @NonNegative
    public int enumerate(Thread[] list);

    @NonNegative
    public int enumerate(Thread[] list, boolean recurse);

    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @SuppressWarnings("index:array.access.unsafe.high")
    @NonNegative
    public int activeGroupCount();

    @NonNegative
    public int enumerate(ThreadGroup[] list);

    @NonNegative
    public int enumerate(ThreadGroup[] list, boolean recurse);

    @Deprecated()
    public final void stop();

    @CFComment({ " groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @SuppressWarnings("index:array.access.unsafe.high")
    public final void interrupt();

    @Deprecated()
    @SuppressWarnings("removal")
    public final void suspend();

    @CFComment({ "index:  // groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @Deprecated()
    @SuppressWarnings({ "removal", "index:array.access.unsafe.high" })
    public final void resume();

    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @SuppressWarnings("index:array.access.unsafe.high")
    @Deprecated()
    public final void destroy();

    void addUnstarted();

    @CFComment({ "index: #1: If nthreads = threads.length, length of threads is doubled" })
    @SuppressWarnings({ "index:array.access.unsafe.high", "index:compound.assignment" })
    void add(Thread t);

    void threadStartFailed(Thread t);

    void threadTerminated(Thread t);

    @CFComment({ "index: groupSnapshot.length = ngroupsSnapshot by #0.1", "for the else case, ngroupsSnapshot will be null and it will never enter the group as nGroups will be 0" })
    @SuppressWarnings("index:array.access.unsafe.high")
    public void list();

    void list(PrintStream out, int indent);

    public void uncaughtException(Thread t, Throwable e);

    @Deprecated()
    public boolean allowThreadSuspension(boolean b);

    @SideEffectFree
    public String toString(@GuardSatisfied ThreadGroup this);
}
