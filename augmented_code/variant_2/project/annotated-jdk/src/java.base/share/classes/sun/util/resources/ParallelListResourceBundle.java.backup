/*
 * Copyright (c) 2013, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.resources;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.util.AbstractSet;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicMarkableReference;

public abstract class ParallelListResourceBundle extends ResourceBundle {

    protected ParallelListResourceBundle() {
    }

    protected abstract Object[][] getContents();

    ResourceBundle getParent();

    public void setParallelContents(OpenListResourceBundle rb);

    boolean areParallelContentsComplete();

    @Override
    protected Object handleGetObject(String key);

    @Override
    public Enumeration<String> getKeys();

    @Override
    @Pure
    public boolean containsKey(@GuardSatisfied @UnknownSignedness String key);

    @Override
    protected Set<String> handleKeySet();

    @Override
    @SuppressWarnings("UnusedAssignment")
    public Set<String> keySet();

    synchronized void resetKeySet();

    void loadLookupTablesIfNecessary();

    private static class KeySet extends AbstractSet<String> {

        @Override
        @Pure
        public boolean contains(@UnknownSignedness Object o);

        @Override
        public Iterator<String> iterator();

        @Override
        public int size();
    }
}
