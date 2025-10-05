/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import java.util.concurrent.TimeUnit;

    @Positive
@AnnotatedFor({ "lock" })
    @Positive
public interface Lock {

    @Positive
    @EnsuresLockHeld({ "this" })
    @Positive
    @ReleasesNoLocks
    @Positive
    void lock();

    @Positive
    @EnsuresLockHeld({ "this" })
    @Positive
    @ReleasesNoLocks
    @Positive
    void lockInterruptibly() throws InterruptedException;

    @Positive
    @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @Positive
    @ReleasesNoLocks
    @Positive
    boolean tryLock();

    @Positive
    @EnsuresLockHeldIf(expression = { "this" }, result = true)
    @Positive
    @ReleasesNoLocks
    @Positive
    boolean tryLock(long time, TimeUnit unit) throws InterruptedException;

    @Positive
    @MayReleaseLocks
    @Positive
    void unlock();

    @Positive
    Condition newCondition();
    @Positive
}
