/*
    @Positive
 * Copyright (c) 2016, Oracle and/or its affiliates. All rights reserved.
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
package jdk.internal.jshell.tool;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.nio.file.AccessDeniedException;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.NoSuchFileException;
    @Positive
import java.time.LocalDateTime;
    @Positive
import java.time.format.DateTimeFormatter;
    @Positive
import java.time.format.FormatStyle;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import static java.util.stream.Collectors.joining;
    @Positive
import static jdk.internal.jshell.tool.JShellTool.RECORD_SEPARATOR;
    @Positive
import static jdk.internal.jshell.tool.JShellTool.getResource;
    @Positive
import static jdk.internal.jshell.tool.JShellTool.readResource;
    @Positive
import static jdk.internal.jshell.tool.JShellTool.toPathResolvingUserHome;

    @Positive
class Startup {

    @Positive
    private static class StartupEntry {

    @Positive
        String storedForm();

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);
    @Positive
    }

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    boolean isEmpty();

    @Positive
    boolean isDefault();

    @Positive
    String storedForm();

    @Positive
    String show(boolean isRetained);

    @Positive
    String showDetail();

    @Positive
    static Startup unpack(String storedForm, MessageHandler mh);

    @Positive
    static Startup fromFileList(List<String> fns, String context, MessageHandler mh);

    @Positive
    static Startup noStartup();

    @Positive
    static Startup defaultStartup(MessageHandler mh);
    @Positive
}

// CFWR semantic augmentation - variant 0
