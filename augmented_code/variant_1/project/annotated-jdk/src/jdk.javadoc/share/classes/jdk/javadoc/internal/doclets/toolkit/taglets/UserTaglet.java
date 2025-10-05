/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.taglets;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import com.sun.source.doctree.DocTree;
    @Positive
import jdk.javadoc.internal.doclets.formats.html.markup.RawHtml;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.Content;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.util.Utils;
    @Positive
import static jdk.javadoc.doclet.Taglet.Location.*;

    @Positive
public final class UserTaglet implements Taglet {

    @Positive
    public UserTaglet(jdk.javadoc.doclet.Taglet t) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public Set<jdk.javadoc.doclet.Taglet.Location> getAllowedLocations();

    @Positive
    @Override
    @Positive
    public boolean inField();

    @Positive
    @Override
    @Positive
    public boolean inConstructor();

    @Positive
    @Override
    @Positive
    public boolean inMethod();

    @Positive
    @Override
    @Positive
    public boolean inOverview();

    @Positive
    @Override
    @Positive
    public boolean inModule();

    @Positive
    @Override
    @Positive
    public boolean inPackage();

    @Positive
    @Override
    @Positive
    public boolean inType();

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean isInlineTag();

    @Positive
    @Override
    @Positive
    public boolean isBlockTag();

    @Positive
    @Override
    @Positive
    public String getName();

    @Positive
    @Override
    @Positive
    public Content getInlineTagOutput(Element element, DocTree tag, TagletWriter writer);

    @Positive
    @Override
    @Positive
    public Content getAllBlockTagOutput(Element holder, TagletWriter writer);
    @Positive
}
