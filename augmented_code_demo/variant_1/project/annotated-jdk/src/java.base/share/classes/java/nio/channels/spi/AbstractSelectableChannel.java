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
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.common.returnsreceiver.qual.This;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.nio.channels.CancelledKeyException;
    @Positive
import java.nio.channels.ClosedChannelException;
    @Positive
import java.nio.channels.ClosedSelectorException;
    @Positive
import java.nio.channels.IllegalBlockingModeException;
    @Positive
import java.nio.channels.IllegalSelectorException;
    @Positive
import java.nio.channels.SelectableChannel;
    @Positive
import java.nio.channels.SelectionKey;
    @Positive
import java.nio.channels.Selector;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.function.Consumer;

    @Positive
@AnnotatedFor({ "mustcall", "returnsreceiver" })
    @Positive
public abstract class AbstractSelectableChannel extends SelectableChannel {

    @Positive
    protected AbstractSelectableChannel(SelectorProvider provider) {
    @Positive
    }

    @Positive
    public final SelectorProvider provider();

    @Positive
    void removeKey(SelectionKey k);

    @Positive
    public final boolean isRegistered();

    @Positive
    public final SelectionKey keyFor(Selector sel);

    @Positive
    public final SelectionKey register(Selector sel, int ops, Object att) throws ClosedChannelException;

    @Positive
    protected final void implCloseChannel() throws IOException;

    @Positive
    protected abstract void implCloseSelectableChannel() throws IOException;

    @Positive
    public final boolean isBlocking();

    @Positive
    public final Object blockingLock();

    @Positive
    @MustCallAlias
    @Positive
    @This
    @Positive
    public final SelectableChannel configureBlocking(@MustCallAlias AbstractSelectableChannel this, boolean block) throws IOException;

    @Positive
    protected abstract void implConfigureBlocking(boolean block) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
