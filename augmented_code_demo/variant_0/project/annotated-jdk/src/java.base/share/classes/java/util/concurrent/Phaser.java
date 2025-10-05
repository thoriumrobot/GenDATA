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
package java.util.concurrent;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.util.concurrent.atomic.AtomicReference;
    @Positive
import java.util.concurrent.locks.LockSupport;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Phaser {

    @Positive
    public Phaser() {
    @Positive
    }

    @Positive
    public Phaser(int parties) {
    @Positive
    }

    @Positive
    public Phaser(Phaser parent) {
    @Positive
    }

    @Positive
    public Phaser(Phaser parent, int parties) {
    @Positive
    }

    @Positive
    public int register();

    @Positive
    public int bulkRegister(int parties);

    @Positive
    public int arrive();

    @Positive
    public int arriveAndDeregister();

    @Positive
    public int arriveAndAwaitAdvance();

    @Positive
    public int awaitAdvance(int phase);

    @Positive
    public int awaitAdvanceInterruptibly(int phase) throws InterruptedException;

    @Positive
    public int awaitAdvanceInterruptibly(int phase, long timeout, TimeUnit unit) throws InterruptedException, TimeoutException;

    @Positive
    public void forceTermination();

    @Positive
    public final int getPhase();

    @Positive
    public int getRegisteredParties();

    @Positive
    public int getArrivedParties();

    @Positive
    public int getUnarrivedParties();

    @Positive
    public Phaser getParent();

    @Positive
    public Phaser getRoot();

    @Positive
    public boolean isTerminated();

    @Positive
    protected boolean onAdvance(int phase, int registeredParties);

    @Positive
    public String toString();

    @Positive
    static final class QNode implements ForkJoinPool.ManagedBlocker {

    @Positive
        public boolean isReleasable();

    @Positive
        public boolean block();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
