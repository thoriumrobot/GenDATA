/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.*;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Date;
import java.util.concurrent.atomic.AtomicInteger;
import java.lang.ref.Cleaner.Cleanable;
import jdk.internal.ref.CleanerFactory;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@UsesObjectEquals
public class Timer {

    private static class ThreadReaper implements Runnable {

        public void run();
    }

    public Timer() {
    }

    public Timer(boolean isDaemon) {
    }

    public Timer(String name) {
    }

    public Timer(String name, boolean isDaemon) {
    }

    public void schedule(TimerTask task, long delay);

    public void schedule(TimerTask task, Date time);

    public void schedule(TimerTask task, long delay, long period);

    public void schedule(TimerTask task, Date firstTime, long period);

    public void scheduleAtFixedRate(TimerTask task, long delay, long period);

    public void scheduleAtFixedRate(TimerTask task, Date firstTime, long period);

    public void cancel();

    @NonNegative
    public int purge();
}

class TimerThread extends Thread {

    public void run();
}

class TaskQueue {

    @Pure
    int size();

    void add(TimerTask task);

    TimerTask getMin();

    TimerTask get(int i);

    void removeMin();

    void quickRemove(int i);

    void rescheduleMin(long newTime);

    @Pure
    @EnsuresNonEmptyIf(result = false, expression = "this")
    boolean isEmpty();

    void clear();

    void heapify();
}
