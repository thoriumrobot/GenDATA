/*
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.nullness.qual.Nullable;
import jdk.internal.reflect.CallerSensitive;
import java.lang.invoke.MethodType;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;

public final class StackWalker {

    public interface StackFrame {

        public String getClassName();

        public String getMethodName();

        public Class<?> getDeclaringClass();

        public default MethodType getMethodType();

        public default String getDescriptor();

        public int getByteCodeIndex();

        public String getFileName();

        public int getLineNumber();

        public boolean isNativeMethod();

        public StackTraceElement toStackTraceElement();
    }

    public enum Option {

        RETAIN_CLASS_REFERENCE, SHOW_REFLECT_FRAMES, SHOW_HIDDEN_FRAMES
    }

    public static StackWalker getInstance();

    public static StackWalker getInstance(Option option);

    public static StackWalker getInstance(Set<Option> options);

    public static StackWalker getInstance(Set<Option> options, int estimateDepth);

    @CallerSensitive
    public <T extends @Nullable Object> T walk(Function<? super Stream<StackFrame>, ? extends T> function);

    @CallerSensitive
    public void forEach(Consumer<? super StackFrame> action);

    @CallerSensitive
    public Class<?> getCallerClass();

    static StackWalker newInstance(Set<Option> options, ExtendedOption extendedOption);

    int estimateDepth();

    boolean hasOption(Option option);

    boolean hasLocalsOperandsOption();
}
