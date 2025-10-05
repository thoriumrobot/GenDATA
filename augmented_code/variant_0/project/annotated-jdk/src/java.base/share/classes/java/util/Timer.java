/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.*;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Date;
    @Positive
import java.util.concurrent.atomic.AtomicInteger;
    @Positive
import java.lang.ref.Cleaner.Cleanable;
    @Positive
import jdk.internal.ref.CleanerFactory;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class Timer {

    @Positive
    private static class ThreadReaper implements Runnable {

    @Positive
        public void run();
    @Positive
    }

    @Positive
    public Timer() {
    @Positive
    }

    @Positive
    public Timer(boolean isDaemon) {
    @Positive
    }

    @Positive
    public Timer(String name) {
    @Positive
    }

    @Positive
    public Timer(String name, boolean isDaemon) {
    @Positive
    }

    @Positive
    public void schedule(TimerTask task, long delay);

    @Positive
    public void schedule(TimerTask task, Date time);

    @Positive
    public void schedule(TimerTask task, long delay, long period);

    @Positive
    public void schedule(TimerTask task, Date firstTime, long period);

    @Positive
    public void scheduleAtFixedRate(TimerTask task, long delay, long period);

    @Positive
    public void scheduleAtFixedRate(TimerTask task, Date firstTime, long period);

    @Positive
    public void cancel();

    @Positive
    @NonNegative
    @Positive
    public int purge();
    @Positive
}

    @Positive
class TimerThread extends Thread {

    @Positive
    public void run();
    @Positive
}

    @Positive
class TaskQueue {

    @Positive
    @Pure
    @Positive
    int size();

    @Positive
    void add(TimerTask task);

    @Positive
    TimerTask getMin();

    @Positive
    TimerTask get(int i);

    @Positive
    void removeMin();

    @Positive
    void quickRemove(int i);

    @Positive
    void rescheduleMin(long newTime);

    @Positive
    @Pure
    @Positive
    @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
    boolean isEmpty();

    @Positive
    void clear();

    @Positive
    void heapify();
    @Positive
}
