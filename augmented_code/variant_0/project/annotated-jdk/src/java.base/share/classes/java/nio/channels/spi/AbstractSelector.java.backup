/*
 * Copyright (c) 2000, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels.spi;

import org.checkerframework.checker.calledmethods.qual.EnsuresCalledMethodsIf;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.util.HashSet;
import java.util.Set;
import sun.nio.ch.Interruptible;
import sun.nio.ch.SelectorImpl;

@AnnotatedFor({ "mustcall" })
public abstract class AbstractSelector extends Selector {

    protected AbstractSelector(SelectorProvider provider) {
    }

    void cancel(SelectionKey k);

    public final void close() throws IOException;

    protected abstract void implCloseSelector() throws IOException;

    @EnsuresCalledMethodsIf(expression = "this", result = false, methods = { "close" })
    public final boolean isOpen();

    public final SelectorProvider provider();

    protected final Set<SelectionKey> cancelledKeys();

    protected abstract SelectionKey register(AbstractSelectableChannel ch, int ops, Object att);

    protected final void deregister(AbstractSelectionKey key);

    protected final void begin();

    protected final void end();
}
