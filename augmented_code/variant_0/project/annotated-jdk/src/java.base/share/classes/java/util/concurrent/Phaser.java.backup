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
package java.util.concurrent;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Phaser {

    public Phaser() {
    }

    public Phaser(int parties) {
    }

    public Phaser(Phaser parent) {
    }

    public Phaser(Phaser parent, int parties) {
    }

    public int register();

    public int bulkRegister(int parties);

    public int arrive();

    public int arriveAndDeregister();

    public int arriveAndAwaitAdvance();

    public int awaitAdvance(int phase);

    public int awaitAdvanceInterruptibly(int phase) throws InterruptedException;

    public int awaitAdvanceInterruptibly(int phase, long timeout, TimeUnit unit) throws InterruptedException, TimeoutException;

    public void forceTermination();

    public final int getPhase();

    public int getRegisteredParties();

    public int getArrivedParties();

    public int getUnarrivedParties();

    public Phaser getParent();

    public Phaser getRoot();

    public boolean isTerminated();

    protected boolean onAdvance(int phase, int registeredParties);

    public String toString();

    static final class QNode implements ForkJoinPool.ManagedBlocker {

        public boolean isReleasable();

        public boolean block();
    }
}
