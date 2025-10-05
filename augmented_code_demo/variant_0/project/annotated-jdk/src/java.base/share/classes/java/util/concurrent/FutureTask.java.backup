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
import java.util.concurrent.locks.LockSupport;

@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public class FutureTask<V> implements RunnableFuture<V> {

    public FutureTask(Callable<V> callable) {
    }

    public FutureTask(Runnable runnable, V result) {
    }

    public boolean isCancelled();

    public boolean isDone();

    public boolean cancel(boolean mayInterruptIfRunning);

    public V get() throws InterruptedException, ExecutionException;

    public V get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException;

    protected void done();

    protected void set(V v);

    protected void setException(Throwable t);

    public void run();

    protected boolean runAndReset();

    static final class WaitNode {
    }

    public String toString();
}
