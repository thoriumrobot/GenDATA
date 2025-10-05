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

@AnnotatedFor({ "lock" })
public class ReentrantReadWriteLock implements ReadWriteLock, java.io.Serializable {

    public ReentrantReadWriteLock() {
    }

    public ReentrantReadWriteLock(boolean fair) {
    }

    public ReentrantReadWriteLock.WriteLock writeLock();

    public ReentrantReadWriteLock.ReadLock readLock();

    abstract static class Sync extends AbstractQueuedSynchronizer {

        static int sharedCount(int c);

        static int exclusiveCount(int c);

        static final class HoldCounter {
        }

        static final class ThreadLocalHoldCounter extends ThreadLocal<HoldCounter> {

            public HoldCounter initialValue();
        }

        abstract boolean readerShouldBlock();

        abstract boolean writerShouldBlock();

        @ReservedStackAccess
        protected final boolean tryRelease(int releases);

        @ReservedStackAccess
        protected final boolean tryAcquire(int acquires);

        @ReservedStackAccess
        protected final boolean tryReleaseShared(int unused);

        @ReservedStackAccess
        protected final int tryAcquireShared(int unused);

        final int fullTryAcquireShared(Thread current);

        @ReservedStackAccess
        final boolean tryWriteLock();

        @ReservedStackAccess
        final boolean tryReadLock();

        protected final boolean isHeldExclusively();

        final ConditionObject newCondition();

        final Thread getOwner();

        final int getReadLockCount();

        final boolean isWriteLocked();

        final int getWriteHoldCount();

        final int getReadHoldCount();

        final int getCount();
    }

    static final class NonfairSync extends Sync {

        final boolean writerShouldBlock();

        final boolean readerShouldBlock();
    }

    static final class FairSync extends Sync {

        final boolean writerShouldBlock();

        final boolean readerShouldBlock();
    }

    public static class ReadLock implements Lock, java.io.Serializable {

        protected ReadLock(ReentrantReadWriteLock lock) {
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

        public String toString();
    }

    public static class WriteLock implements Lock, java.io.Serializable {

        protected WriteLock(ReentrantReadWriteLock lock) {
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

        public String toString();

        @EnsuresLockHeldIf(expression = { "this" }, result = true)
        @ReleasesNoLocks
        public boolean isHeldByCurrentThread();

        public int getHoldCount();
    }

    public final boolean isFair();

    protected Thread getOwner();

    public int getReadLockCount();

    public boolean isWriteLocked();

    public boolean isWriteLockedByCurrentThread();

    public int getWriteHoldCount();

    public int getReadHoldCount();

    protected Collection<Thread> getQueuedWriterThreads();

    protected Collection<Thread> getQueuedReaderThreads();

    public final boolean hasQueuedThreads();

    public final boolean hasQueuedThread(Thread thread);

    public final int getQueueLength();

    protected Collection<Thread> getQueuedThreads();

    public boolean hasWaiters(Condition condition);

    public int getWaitQueueLength(Condition condition);

    protected Collection<Thread> getWaitingThreads(Condition condition);

    public String toString();
}
