/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package java.awt.datatransfer;

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
import java.io.ByteArrayInputStream;
    @Positive
import java.io.CharArrayReader;
    @Positive
import java.io.Externalizable;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.io.ObjectInput;
    @Positive
import java.io.ObjectOutput;
    @Positive
import java.io.OptionalDataException;
    @Positive
import java.io.Reader;
    @Positive
import java.io.Serial;
    @Positive
import java.io.StringReader;
    @Positive
import java.io.UnsupportedEncodingException;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.CharBuffer;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Objects;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import sun.datatransfer.DataFlavorUtil;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
public class DataFlavor implements Externalizable, Cloneable {

    @Positive
    protected static final Class<?> tryToLoadClass(String className, ClassLoader fallback) throws ClassNotFoundException;

    @Positive
    public static final DataFlavor stringFlavor;

    @Positive
    public static final DataFlavor imageFlavor;

    @Positive
    @Deprecated
    @Positive
    public static final DataFlavor plainTextFlavor;

    @Positive
    @Interned
    @Positive
    public static final String javaSerializedObjectMimeType;

    @Positive
    public static final DataFlavor javaFileListFlavor;

    @Positive
    @Interned
    @Positive
    public static final String javaJVMLocalObjectMimeType;

    @Positive
    @Interned
    @Positive
    public static final String javaRemoteObjectMimeType;

    @Positive
    public static final DataFlavor selectionHtmlFlavor;

    @Positive
    public static final DataFlavor fragmentHtmlFlavor;

    @Positive
    public static final DataFlavor allHtmlFlavor;

    @Positive
    public DataFlavor() {
    @Positive
    }

    @Positive
    public DataFlavor(Class<?> representationClass, String humanPresentableName) {
    @Positive
    }

    @Positive
    public DataFlavor(String mimeType, String humanPresentableName) {
    @Positive
    }

    @Positive
    public DataFlavor(String mimeType, String humanPresentableName, ClassLoader classLoader) throws ClassNotFoundException {
    @Positive
    }

    @Positive
    public DataFlavor(String mimeType) throws ClassNotFoundException {
    @Positive
    }

    @Positive
    public String toString();

    @Positive
    public static final DataFlavor getTextPlainUnicodeFlavor();

    @Positive
    public static final DataFlavor selectBestTextFlavor(DataFlavor[] availableFlavors);

    @Positive
    public Reader getReaderForText(Transferable transferable) throws UnsupportedFlavorException, IOException;

    @Positive
    public String getMimeType();

    @Positive
    public Class<?> getRepresentationClass();

    @Positive
    public String getHumanPresentableName();

    @Positive
    public String getPrimaryType();

    @Positive
    public String getSubType();

    @Positive
    public String getParameter(String paramName);

    @Positive
    public void setHumanPresentableName(String humanPresentableName);

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public boolean equals(DataFlavor that);

    @Positive
    @Deprecated
    @Positive
    public boolean equals(String s);

    @Positive
    public int hashCode();

    @Positive
    public boolean match(DataFlavor that);

    @Positive
    public boolean isMimeTypeEqual(String mimeType);

    @Positive
    public final boolean isMimeTypeEqual(DataFlavor dataFlavor);

    @Positive
    public boolean isMimeTypeSerializedObject();

    @Positive
    public final Class<?> getDefaultRepresentationClass();

    @Positive
    public final String getDefaultRepresentationClassAsString();

    @Positive
    public boolean isRepresentationClassInputStream();

    @Positive
    public boolean isRepresentationClassReader();

    @Positive
    public boolean isRepresentationClassCharBuffer();

    @Positive
    public boolean isRepresentationClassByteBuffer();

    @Positive
    public boolean isRepresentationClassSerializable();

    @Positive
    public boolean isRepresentationClassRemote();

    @Positive
    public boolean isFlavorSerializedObjectType();

    @Positive
    public boolean isFlavorRemoteObjectType();

    @Positive
    public boolean isFlavorJavaFileListType();

    @Positive
    public boolean isFlavorTextType();

    @Positive
    public synchronized void writeExternal(ObjectOutput os) throws IOException;

    @Positive
    public synchronized void readExternal(ObjectInput is) throws IOException, ClassNotFoundException;

    @Positive
    public Object clone() throws CloneNotSupportedException;

    @Positive
    @Deprecated
    @Positive
    protected String normalizeMimeTypeParameter(String parameterName, String parameterValue);

    @Positive
    @Deprecated
    @Positive
    protected String normalizeMimeType(String mimeType);
    @Positive
}
