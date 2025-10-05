/*
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
package java.util.concurrent.locks;

    @Positive
import org.checkerframework.checker.lock.qual.EnsuresLockHeld;
    @Positive
import org.checkerframework.checker.lock.qual.EnsuresLockHeldIf;
    @Positive
import org.checkerframework.checker.lock.qual.MayReleaseLocks;
    @Positive
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Collection;
    @Positive
import java.util.concurrent.TimeUnit;
    @Positive
import jdk.internal.vm.annotation.ReservedStackAccess;

    @Positive
@AnnotatedFor({ "lock" })
    @Positive
public class ReentrantReadWriteLock implements ReadWriteLock, java.io.Serializable {

    @Positive
    public ReentrantReadWriteLock() {
    @Positive
    }

    @Positive
    public ReentrantReadWriteLock(boolean fair) {
    @Positive
    }

    @Positive
    public ReentrantReadWriteLock.WriteLock writeLock();

    @Positive
    public ReentrantReadWriteLock.ReadLock readLock();

    @Positive
    abstract static class Sync extends AbstractQueuedSynchronizer {

    @Positive
        static int sharedCount(int c);

    @Positive
        static int exclusiveCount(int c);

    @Positive
        static final class HoldCounter {
    @Positive
        }

    @Positive
        static final class ThreadLocalHoldCounter extends ThreadLocal<HoldCounter> {

    @Positive
            public HoldCounter initialValue();
    @Positive
        }

    @Positive
        abstract boolean readerShouldBlock();

    @Positive
        abstract boolean writerShouldBlock();

    @Positive
        @ReservedStackAccess
    @Positive
        protected final boolean tryRelease(int releases);

    @Positive
        @ReservedStackAccess
    @Positive
        protected final boolean tryAcquire(int acquires);

    @Positive
        @ReservedStackAccess
    @Positive
        protected final boolean tryReleaseShared(int unused);

    @Positive
        @ReservedStackAccess
    @Positive
        protected final int tryAcquireShared(int unused);

    @Positive
        final int fullTryAcquireShared(Thread current);

    @Positive
        @ReservedStackAccess
    @Positive
        final boolean tryWriteLock();

    @Positive
        @ReservedStackAccess
    @Positive
        final boolean tryReadLock();

    @Positive
        protected final boolean isHeldExclusively();

    @Positive
        final ConditionObject newCondition();

    @Positive
        final Thread getOwner();

    @Positive
        final int getReadLockCount();

    @Positive
        final boolean isWriteLocked();

    @Positive
        final int getWriteHoldCount();

    @Positive
        final int getReadHoldCount();

    @Positive
        final int getCount();
    @Positive
    }

    @Positive
    static final class NonfairSync extends Sync {

    @Positive
        final boolean writerShouldBlock();

    @Positive
        final boolean readerShouldBlock();
    @Positive
    }

    @Positive
    static final class FairSync extends Sync {

    @Positive
        final boolean writerShouldBlock();

    @Positive
        final boolean readerShouldBlock();
    @Positive
    }

    @Positive
    public static class ReadLock implements Lock, java.io.Serializable {

    @Positive
        protected ReadLock(ReentrantReadWriteLock lock) {
    @Positive
        }

    @Positive
        @EnsuresLockHeld({ "this" })
    @Positive
        @ReleasesNoLocks
    @Positive
        public void lock();

    @Positive
        @EnsuresLockHeld({ "this" })
    @Positive
        @ReleasesNoLocks
    @Positive
        public void lockInterruptibly() throws InterruptedException;

    @Positive
        @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @Positive
        @ReleasesNoLocks
    @Positive
        public boolean tryLock();

    @Positive
        @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @Positive
        @ReleasesNoLocks
    @Positive
        public boolean tryLock(long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
        @MayReleaseLocks
    @Positive
        public void unlock();

    @Positive
        public Condition newCondition();

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static class WriteLock implements Lock, java.io.Serializable {

    @Positive
        protected WriteLock(ReentrantReadWriteLock lock) {
    @Positive
        }

    @Positive
        @EnsuresLockHeld({ "this" })
    @Positive
        @ReleasesNoLocks
    @Positive
        public void lock();

    @Positive
        @EnsuresLockHeld({ "this" })
    @Positive
        @ReleasesNoLocks
    @Positive
        public void lockInterruptibly() throws InterruptedException;

    @Positive
        @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @Positive
        @ReleasesNoLocks
    @Positive
        public boolean tryLock();

    @Positive
        @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @Positive
        @ReleasesNoLocks
    @Positive
        public boolean tryLock(long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
        @MayReleaseLocks
    @Positive
        public void unlock();

    @Positive
        public Condition newCondition();

    @Positive
        public String toString();

    @Positive
        @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @Positive
        @ReleasesNoLocks
    @Positive
        public boolean isHeldByCurrentThread();

    @Positive
        public int getHoldCount();
    @Positive
    }

    @Positive
    public final boolean isFair();

    @Positive
    protected Thread getOwner();

    @Positive
    public int getReadLockCount();

    @Positive
    public boolean isWriteLocked();

    @Positive
    public boolean isWriteLockedByCurrentThread();

    @Positive
    public int getWriteHoldCount();

    @Positive
    public int getReadHoldCount();

    @Positive
    protected Collection<Thread> getQueuedWriterThreads();

    @Positive
    protected Collection<Thread> getQueuedReaderThreads();

    @Positive
    public final boolean hasQueuedThreads();

    @Positive
    public final boolean hasQueuedThread(Thread thread);

    @Positive
    public final int getQueueLength();

    @Positive
    protected Collection<Thread> getQueuedThreads();

    @Positive
    public boolean hasWaiters(Condition condition);

    @Positive
    public int getWaitQueueLength(Condition condition);

    @Positive
    protected Collection<Thread> getWaitingThreads(Condition condition);

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
