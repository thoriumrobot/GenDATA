/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.security;

    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import java.io.*;
    @Positive
import java.util.*;
    @Positive
import static java.util.Locale.ENGLISH;
    @Positive
import java.lang.ref.*;
    @Positive
import java.lang.reflect.*;
    @Positive
import java.util.function.BiConsumer;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.concurrent.ConcurrentHashMap;

    @Positive
public abstract class Provider extends Properties {

    @Positive
    @Deprecated()
    @Positive
    protected Provider(String name, double version, String info) {
    @Positive
    }

    @Positive
    protected Provider(String name, String versionStr, String info) {
    @Positive
    }

    @Positive
    public Provider configure(String configArg);

    @Positive
    public boolean isConfigured();

    @Positive
    public String getName();

    @Positive
    @Deprecated()
    @Positive
    public double getVersion();

    @Positive
    public String getVersionStr();

    @Positive
    public String getInfo();

    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public synchronized void clear();

    @Positive
    @Override
    @Positive
    public synchronized void load(InputStream inStream) throws IOException;

    @Positive
    @Override
    @Positive
    public synchronized void putAll(Map<?, ?> t);

    @Positive
    @Override
    @Positive
    public synchronized Set<Map.Entry<Object, Object>> entrySet();

    @Positive
    @Override
    @Positive
    public Set<Object> keySet();

    @Positive
    @Override
    @Positive
    public Collection<Object> values();

    @Positive
    @Override
    @Positive
    public synchronized Object put(Object key, Object value);

    @Positive
    @Override
    @Positive
    public synchronized Object putIfAbsent(Object key, Object value);

    @Positive
    @Override
    @Positive
    public synchronized Object remove(Object key);

    @Positive
    @Override
    @Positive
    public synchronized boolean remove(@UnknownSignedness Object key, @UnknownSignedness Object value);

    @Positive
    @Override
    @Positive
    public synchronized boolean replace(Object key, Object oldValue, Object newValue);

    @Positive
    @Override
    @Positive
    public synchronized Object replace(Object key, Object value);

    @Positive
    @Override
    @Positive
    public synchronized void replaceAll(BiFunction<? super Object, ? super Object, ? extends Object> function);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized Object compute(Object key, BiFunction<? super Object, ? super Object, ? extends @PolyNull Object> remappingFunction);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized Object computeIfAbsent(Object key, Function<? super Object, ? extends @PolyNull Object> mappingFunction);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized Object computeIfPresent(Object key, BiFunction<? super Object, ? super Object, ? extends @PolyNull Object> remappingFunction);

    @Positive
    @Override
    @Positive
    @PolyNull
    @Positive
    public synchronized Object merge(Object key, Object value, BiFunction<? super Object, ? super Object, ? extends @PolyNull Object> remappingFunction);

    @Positive
    @Override
    @Positive
    public Object get(Object key);

    @Positive
    @Override
    @Positive
    public synchronized Object getOrDefault(Object key, Object defaultValue);

    @Positive
    @Override
    @Positive
    public synchronized void forEach(BiConsumer<? super Object, ? super Object> action);

    @Positive
    @Override
    @Positive
    public Enumeration<Object> keys();

    @Positive
    @Override
    @Positive
    public Enumeration<Object> elements();

    @Positive
    public String getProperty(String key);

    @Positive
    private static class ServiceKey {

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);

    @Positive
        boolean matches(String type, String algorithm);
    @Positive
    }

    @Positive
    public Service getService(String type, String algorithm);

    @Positive
    public synchronized Set<Service> getServices();

    @Positive
    protected void putService(Service s);

    @Positive
    synchronized Service getDefaultSecureRandomService();

    @Positive
    protected void removeService(Service s);

    @Positive
    private static class UString {

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static class EngineDescription {

    @Positive
        Class<?> getConstructorParameterClass() throws ClassNotFoundException;
    @Positive
    }

    @Positive
    public static class Service {

    @Positive
        void addAttribute(String type, String value);

    @Positive
        public Service(Provider provider, String type, String algorithm, String className, List<String> aliases, Map<String, String> attributes) {
    @Positive
        }

    @Positive
        public final String getType();

    @Positive
        public final String getAlgorithm();

    @Positive
        public final Provider getProvider();

    @Positive
        public final String getClassName();

    @Positive
        public final String getAttribute(String name);

    @Positive
        public Object newInstance(Object constructorParameter) throws NoSuchAlgorithmException;

    @Positive
        public boolean supportsParameter(Object parameter);

    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
