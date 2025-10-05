/*
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
package java.util.concurrent.locks;

import org.checkerframework.checker.lock.qual.EnsuresLockHeld;
import org.checkerframework.checker.lock.qual.EnsuresLockHeldIf;
import org.checkerframework.checker.lock.qual.MayReleaseLocks;
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.Collection;
import java.util.concurrent.TimeUnit;
import jdk.internal.vm.annotation.ReservedStackAccess;

@AnnotatedFor("lock")
public class ReentrantLock implements Lock, java.io.Serializable {

    abstract static class Sync extends AbstractQueuedSynchronizer {

        @ReservedStackAccess
        final boolean tryLock();

        abstract boolean initialTryLock();

        @ReservedStackAccess
        final void lock();

        @ReservedStackAccess
        final void lockInterruptibly() throws InterruptedException;

        @ReservedStackAccess
        final boolean tryLockNanos(long nanos) throws InterruptedException;

        @ReservedStackAccess
        protected final boolean tryRelease(int releases);

        protected final boolean isHeldExclusively();

        final ConditionObject newCondition();

        final Thread getOwner();

        final int getHoldCount();

        final boolean isLocked();
    }

    static final class NonfairSync extends Sync {

        final boolean initialTryLock();

        protected final boolean tryAcquire(int acquires);
    }

    static final class FairSync extends Sync {

        final boolean initialTryLock();

        protected final boolean tryAcquire(int acquires);
    }

    public ReentrantLock() {
    }

    public ReentrantLock(boolean fair) {
    }

    @EnsuresLockHeld({ "this" })
    @ReleasesNoLocks
    public void lock();

    @EnsuresLockHeld({ "this" })
    @ReleasesNoLocks
    public void lockInterruptibly() throws InterruptedException;

    @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @ReleasesNoLocks
    public boolean tryLock();

    @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @ReleasesNoLocks
    public boolean tryLock(long timeout, TimeUnit unit) throws InterruptedException;

    @MayReleaseLocks
    public void unlock();

    public Condition newCondition();

    public int getHoldCount();

    @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @ReleasesNoLocks
    public boolean isHeldByCurrentThread();

    public boolean isLocked();

    public final boolean isFair();

    protected Thread getOwner();

    public final boolean hasQueuedThreads();

    public final boolean hasQueuedThread(Thread thread);

    public final int getQueueLength();

    protected Collection<Thread> getQueuedThreads();

    public boolean hasWaiters(Condition condition);

    public int getWaitQueueLength(Condition condition);

    protected Collection<Thread> getWaitingThreads(Condition condition);

    public String toString();
}
