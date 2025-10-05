/*
 * Copyright (c) 1994, 2016, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;

@CFComment({ "guieffect:", "@PolyUIType class Observable {", "@SafeEffect void addObserver(@PolyUI Observable this, @PolyUI Observer o);", "@SafeEffect void deleteObserver(@PolyUI Observable this, @PolyUI Observer o);", "@PolyUIEffect void notifyObservers(@PolyUI Observable this);", "@PolyUIEffect void notifyObservers(@PolyUI Observable this, Object arg);}" })
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@Deprecated()
@UsesObjectEquals
public class Observable {

    public Observable() {
    }

    public synchronized void addObserver(@GuardSatisfied Observable this, Observer o);

    public synchronized void deleteObserver(@GuardSatisfied Observable this, @Nullable Observer o);

    public void notifyObservers();

    public void notifyObservers(@Nullable Object arg);

    public synchronized void deleteObservers(@GuardSatisfied Observable this);

    protected synchronized void setChanged();

    protected synchronized void clearChanged();

    public synchronized boolean hasChanged();

    @NonNegative
    public synchronized int countObservers();
}
