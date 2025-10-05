/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.net.httpserver;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.*;
    @Positive
import java.util.function.BiFunction;
    @Positive
import com.sun.net.httpserver.*;

    @Positive
public class UnmodifiableHeaders extends Headers {

    @Positive
    public UnmodifiableHeaders(Headers headers) {
    @Positive
    }

    @Positive
    public int size();

    @Positive
    public boolean isEmpty();

    @Positive
    @Pure
    @Positive
    public boolean containsKey(Object key);

    @Positive
    @Pure
    @Positive
    public boolean containsValue(Object value);

    @Positive
    public List<String> get(Object key);

    @Positive
    public String getFirst(String key);

    @Positive
    public List<String> put(String key, List<String> value);

    @Positive
    public void add(String key, String value);

    @Positive
    public void set(String key, String value);

    @Positive
    public List<String> remove(Object key);

    @Positive
    public void putAll(Map<? extends String, ? extends List<String>> t);

    @Positive
    public void clear();

    @Positive
    public Set<String> keySet();

    @Positive
    public Collection<List<String>> values();

    @Positive
    public Set<Map.Entry<String, List<String>>> entrySet();

    @Positive
    public List<String> replace(String key, List<String> value);

    @Positive
    public boolean replace(String key, List<String> oldValue, List<String> newValue);

    @Positive
    public void replaceAll(BiFunction<? super String, ? super List<String>, ? extends List<String>> function);

    @Positive
    public boolean equals(Object o);

    @Positive
    public int hashCode();
    @Positive
}
