/*
    @Positive
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import java.lang.invoke.MethodType;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.stream.Stream;

    @Positive
public final class StackWalker {

    @Positive
    public interface StackFrame {

    @Positive
        public String getClassName();

    @Positive
        public String getMethodName();

    @Positive
        public Class<?> getDeclaringClass();

    @Positive
        public default MethodType getMethodType();

    @Positive
        public default String getDescriptor();

    @Positive
        public int getByteCodeIndex();

    @Positive
        public String getFileName();

    @Positive
        public int getLineNumber();

    @Positive
        public boolean isNativeMethod();

    @Positive
        public StackTraceElement toStackTraceElement();
    @Positive
    }

    @Positive
    public enum Option {

    @Positive
        RETAIN_CLASS_REFERENCE, SHOW_REFLECT_FRAMES, SHOW_HIDDEN_FRAMES
    @Positive
    }

    @Positive
    public static StackWalker getInstance();

    @Positive
    public static StackWalker getInstance(Option option);

    @Positive
    public static StackWalker getInstance(Set<Option> options);

    @Positive
    public static StackWalker getInstance(Set<Option> options, int estimateDepth);

    @Positive
    @CallerSensitive
    @Positive
    public <T extends @Nullable Object> T walk(Function<? super Stream<StackFrame>, ? extends T> function);

    @Positive
    @CallerSensitive
    @Positive
    public void forEach(Consumer<? super StackFrame> action);

    @Positive
    @CallerSensitive
    @Positive
    public Class<?> getCallerClass();

    @Positive
    static StackWalker newInstance(Set<Option> options, ExtendedOption extendedOption);

    @Positive
    int estimateDepth();

    @Positive
    boolean hasOption(Option option);

    @Positive
    boolean hasLocalsOperandsOption();
    @Positive
}

// CFWR semantic augmentation - variant 0
