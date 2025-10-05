/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels.spi;

    @Positive
import org.checkerframework.checker.calledmethods.qual.EnsuresCalledMethodsIf;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.invoke.VarHandle;
    @Positive
import java.nio.channels.SelectionKey;
    @Positive
import java.nio.channels.Selector;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Set;
    @Positive
import sun.nio.ch.Interruptible;
    @Positive
import sun.nio.ch.SelectorImpl;

    @Positive
@AnnotatedFor({ "mustcall" })
    @Positive
public abstract class AbstractSelector extends Selector {

    @Positive
    protected AbstractSelector(SelectorProvider provider) {
    @Positive
    }

    @Positive
    void cancel(SelectionKey k);

    @Positive
    public final void close() throws IOException;

    @Positive
    protected abstract void implCloseSelector() throws IOException;

    @Positive
    @EnsuresCalledMethodsIf(expression = "this", result = false, methods = { "close" })
    @Positive
    public final boolean isOpen();

    @Positive
    public final SelectorProvider provider();

    @Positive
    protected final Set<SelectionKey> cancelledKeys();

    @Positive
    protected abstract SelectionKey register(AbstractSelectableChannel ch, int ops, Object att);

    @Positive
    protected final void deregister(AbstractSelectionKey key);

    @Positive
    protected final void begin();

    @Positive
    protected final void end();
    @Positive
}
