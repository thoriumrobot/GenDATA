/*
    @Positive
 * Copyright (c) 2005, 2013, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.resources;

    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.Set;
    @Positive
import sun.util.ResourceBundleEnumeration;

    @Positive
@AnnotatedFor({ "index" })
    @Positive
public abstract class OpenListResourceBundle extends ResourceBundle {

    @Positive
    protected OpenListResourceBundle() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    protected Object handleGetObject(String key);

    @Positive
    @Override
    @Positive
    public Enumeration<String> getKeys();

    @Positive
    @Override
    @Positive
    protected Set<String> handleKeySet();

    @Positive
    @Override
    @Positive
    public Set<String> keySet();

    @Positive
    protected abstract Object[][] getContents();

    @Positive
    void loadLookupTablesIfNecessary();

    @Positive
    protected <K, V> Map<K, V> createMap(int size);

    @Positive
    protected <E> Set<E> createSet();
    @Positive
}

// CFWR semantic augmentation - variant 0
